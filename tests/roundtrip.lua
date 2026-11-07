-- Batch test: aseprite -b --script-param bridge=<command> [--script-param backend=comfyui]
--             [--script-param mode=indexed] [--script-param variants=<n>]
--             [--script-param model=sd15-inpaint]
--             --script tests/roundtrip.lua

local script_dir = app.fs.filePath(debug.getinfo(1, "S").source:sub(2))
local inpaint = dofile(app.fs.joinPath(script_dir, "..", "extension", "inpaint.lua"))

local RED = app.pixelColor.rgba(200, 10, 30, 255)
local INVERTED_RED = app.pixelColor.rgba(55, 245, 225, 255)

local function make_sprite(mode)
  if mode == "indexed" then
    local sprite = Sprite(8, 8, ColorMode.INDEXED)
    local palette = sprite.palettes[1]
    palette:resize(4)
    palette:setColor(0, Color{ r = 0, g = 0, b = 0, a = 0 })
    palette:setColor(1, Color{ r = 200, g = 10, b = 30, a = 255 })
    palette:setColor(2, Color{ r = 55, g = 245, b = 225, a = 255 })
    palette:setColor(3, Color{ r = 20, g = 40, b = 220, a = 255 })
    local background = Image(8, 8, ColorMode.INDEXED)
    background:clear(1)
    sprite:newCel(sprite.layers[1], 1, background, Point(0, 0))
    return sprite
  end
  local sprite = Sprite(8, 8, ColorMode.RGB)
  local background = Image(8, 8, ColorMode.RGB)
  background:clear(RED)
  sprite:newCel(sprite.layers[1], 1, background, Point(0, 0))
  return sprite
end

local mode = app.params.mode or "rgb"
local sprite = make_sprite(mode)
sprite.selection = Selection(Rectangle(2, 2, 3, 3))

local backend = app.params.backend or "invert"
local variants = tonumber(app.params.variants or "2")
local options = {
  bridge_command = app.params.bridge,
  margin = 2,
  backend = backend,
  model = app.params.model,
  prompt = "a blue gem",
  denoise = 1.0,
  use_palette = true,
  variants = variants,
}
local job, message = inpaint.generate(sprite, sprite.frames[1], options)
assert(job, message)

-- Background run, polled in a loop since no timer ticks in batch mode.
local function wait_run(run)
  for _ = 1, 1200 do
    local finished, result = inpaint.poll(run)
    if finished ~= nil then
      return finished, result
    end
    os.execute("sleep 0.1")
  end
  error("the background run did not finish")
end

local run = assert(inpaint.start(sprite, sprite.frames[1], options))
local finished, async_job = wait_run(run)
assert(finished, async_job)
assert(#async_job.patches == variants, "the background run returned the wrong variant count")
if backend == "invert" then
  assert(async_job.patches[1]:isEqual(job.patches[1]), "background and blocking runs differ")
end
assert(not app.fs.isFile(run.paths.input), "the background run left its files")

local slow_bridge = app.fs.joinPath(app.fs.tempPath, "aseprite-ai-inpaint-slow-bridge.sh")
local file = assert(io.open(slow_bridge, "w"))
file:write("#!/bin/sh\nexec sleep 30\n")
file:close()
os.execute("chmod +x '" .. slow_bridge .. "'")
local slow_options = {}
for key, value in pairs(options) do
  slow_options[key] = value
end
slow_options.bridge_command = slow_bridge
local started = os.time()
local slow_run = assert(inpaint.start(sprite, sprite.frames[1], slow_options))
inpaint.cancel(slow_run)
local cancelled, cancel_message = wait_run(slow_run)
assert(cancelled == false and cancel_message == nil, "a cancelled run must end without a job")
assert(os.time() - started < 10, "cancel did not stop the bridge")
assert(not app.fs.isFile(slow_run.paths.input), "the cancelled run left its files")

local closing_sprite = Sprite(sprite.width, sprite.height, sprite.colorMode)
closing_sprite.selection = Selection(Rectangle(2, 2, 3, 3))
local closing_run = assert(inpaint.start(closing_sprite, closing_sprite.frames[1], slow_options))
closing_sprite:close()
local closed_finished, closed_message = wait_run(closing_run)
assert(closed_finished == false and closed_message, "closing the sprite must end the run")
assert(not app.fs.isFile(closing_run.paths.input), "the run of a closed sprite left its files")
app.sprite = sprite
os.remove(slow_bridge)
assert(#job.patches == variants, "expected " .. variants .. " patches, got " .. #job.patches)
assert(#job.previews == variants, "expected " .. variants .. " previews")
if backend == "invert" then
  assert(inpaint.changed_fraction(job, 1) == 1, "invert must change every selected pixel")
end
local untouched = { region = job.region, mask = job.mask, original = job.original,
                    previews = { job.original } }
assert(inpaint.changed_fraction(untouched, 1) == 0, "the original must count as unchanged")
inpaint.apply_new_layer(sprite, sprite.frames[1], job, variants)

local top = sprite.layers[#sprite.layers]
assert(top.name == "AI inpaint", "unexpected top layer: " .. top.name)
assert(top:cel(1).image.colorMode == sprite.colorMode, "patch color mode differs")

local flat = Image(8, 8, ColorMode.RGB)
flat:drawSprite(sprite, 1)
if backend == "invert" then
  assert(flat:getPixel(3, 3) == INVERTED_RED, "selected pixel was not rewritten")
  assert(flat:getPixel(4, 4) == INVERTED_RED, "selection corner was not rewritten")
end
assert(flat:getPixel(1, 1) == RED, "pixel outside the selection was modified")
assert(flat:getPixel(5, 5) == RED, "pixel outside the selection was modified")

inpaint.apply_all_layers(sprite, sprite.frames[1], job)
local group = sprite.layers[#sprite.layers]
assert(group.isGroup and group.name == "AI inpaint", "variants group missing")
assert(#group.layers == variants, "expected one layer per variant")
for i, layer in ipairs(group.layers) do
  local number = variants - i + 1
  assert(layer.name == "Variant " .. number, "unexpected variant order: " .. layer.name)
  assert(layer.isVisible == (number == 1), "only the first variant must be visible")
end
assert(group.layers[variants]:cel(1).image:isEqual(job.patches[1]),
       "the visible layer does not hold the first variant")
local with_group = Image(8, 8, ColorMode.RGB)
with_group:drawSprite(sprite, 1)
assert(with_group:getPixel(1, 1) == RED, "the variants group changed a pixel outside the selection")

-- Active layer output: a transparent selected pixel erases the original.
local base = sprite.layers[1]
local indexed = mode == "indexed"
local transparent = indexed and sprite.transparentColor or 0
local red_value = indexed and 1 or RED
local erase_job = { region = job.region, mask = job.mask, patches = { job.patches[1]:clone() } }
erase_job.patches[1]:putPixel(3 - job.region.x, 3 - job.region.y, transparent)

local function layer_pixel(x, y)
  local cel = base:cel(1)
  return cel.image:getPixel(x - cel.position.x, y - cel.position.y)
end

assert(inpaint.active_layer_problem(group) ~= nil, "a group must be refused")
assert(inpaint.active_layer_problem(base) == nil, inpaint.active_layer_problem(base))
inpaint.apply_active_layer(sprite, 1, erase_job, 1, base)
assert(layer_pixel(3, 3) == transparent, "a transparent pixel did not erase the original")
assert(layer_pixel(2, 2) == job.patches[1]:getPixel(2 - job.region.x, 2 - job.region.y),
       "selected pixel not written")
assert(layer_pixel(1, 1) == red_value, "pixel outside the selection was modified")

app.activeLayer = base
app.command.BackgroundFromLayer()
assert(base.isBackground, "the layer was not turned into a background")
app.bgColor = indexed and Color{ index = 3 } or Color{ r = 20, g = 40, b = 220, a = 255 }
local background = indexed and 3 or app.pixelColor.rgba(20, 40, 220, 255)
inpaint.apply_active_layer(sprite, 1, erase_job, 1, base)
assert(layer_pixel(3, 3) == background, "transparency on a background must become bgColor")

base.isEditable = false
assert(inpaint.active_layer_problem(base) ~= nil, "a locked layer must be refused")

print("ROUNDTRIP OK (" .. mode .. ", " .. backend .. ", " .. (app.params.model or "default model")
      .. ", " .. variants .. " variants)")
