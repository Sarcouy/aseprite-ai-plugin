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
local function wait_run(run, seconds)
  for _ = 1, (seconds or 120) * 10 do
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

-- Character task: drawn from the prompt alone, centered on the canvas.
local function palette_values(palette)
  local values = {}
  for i = 0, #palette - 1 do
    local color = palette:getColor(i)
    values[app.pixelColor.rgba(color.red, color.green, color.blue, 255)] = true
  end
  return values
end

local function assert_in_palette(image, palette, what)
  local values = palette_values(palette)
  local opaque = 0
  for pixel in image:pixels() do
    local value = pixel()
    if app.pixelColor.rgbaA(value) > 0 then
      opaque = opaque + 1
      assert(values[value], what .. ": a color is not in the palette")
    end
  end
  assert(opaque > 0, what .. ": the character is empty")
end

local character_sprite = make_sprite(mode)
if mode ~= "indexed" then
  -- A new RGB sprite in batch mode has a single black color, which the bridge ignores.
  local palette = Palette(4)
  palette:setColor(0, Color{ r = 0, g = 0, b = 0, a = 0 })
  palette:setColor(1, Color{ r = 200, g = 10, b = 30, a = 255 })
  palette:setColor(2, Color{ r = 230, g = 140, b = 40, a = 255 })
  palette:setColor(3, Color{ r = 20, g = 40, b = 220, a = 255 })
  character_sprite:setPalette(palette)
end
local character_options = {
  bridge_command = app.params.bridge,
  backend = backend,
  task = "character",
  prompt = "an orange cat",
  size = Size(8, 8),
  use_palette = true,
  variants = variants,
}
local character_job = assert(inpaint.generate(character_sprite, character_sprite.frames[1],
                                              character_options))
assert(character_job.region == Rectangle(0, 0, 8, 8), "the character is not centered")
assert(#character_job.patches == variants, "the character run returned the wrong variant count")
for i, preview in ipairs(character_job.previews) do
  assert_in_palette(preview, character_sprite.palettes[1], "character variant " .. i)
end
if backend == "invert" then
  assert(app.pixelColor.rgbaA(character_job.previews[1]:getPixel(0, 0)) == 0,
         "the character background is not transparent")
  assert(inpaint.changed_fraction(character_job, 1) > 0, "a drawn character counts as empty")
end
inpaint.apply_new_layer(character_sprite, character_sprite.frames[1], character_job, 1)
local character_layer = character_sprite.layers[#character_sprite.layers]
assert(character_layer.name == "AI character", "unexpected layer: " .. character_layer.name)
assert(character_layer:cel(1).image.colorMode == character_sprite.colorMode,
       "character color mode differs")

-- New sprite, with the palette of the active sprite or, without one, its own.
character_options.new_sprite = true
local lent_run = assert(inpaint.start(character_sprite, character_sprite.frames[1],
                                      character_options))
assert(lent_run.sprite == nil, "a new sprite run must not depend on the active sprite")
local lent_finished, lent_job = wait_run(lent_run)
assert(lent_finished, lent_job)
local lent = inpaint.apply_new_sprite(lent_job, 1)
assert(lent.width == 8 and lent.height == 8, "the new sprite has the wrong size")
assert(#lent.palettes[1] == #character_sprite.palettes[1], "the palette was not lent")
assert(#lent.layers == 1 and lent.layers[1].name == "AI character", "unexpected layers")

local alone_run = assert(inpaint.start(nil, nil, character_options))
local alone_finished, alone_job = wait_run(alone_run)
assert(alone_finished, alone_job)
assert(not app.fs.isFile(alone_run.paths.log), "the character run left its files")
local alone = inpaint.apply_new_sprite(alone_job, nil)
assert(#alone.layers == 1 and alone.layers[1].isGroup, "expected one group of variants")
assert(#alone.layers[1].layers == variants, "expected one layer per variant")
assert_in_palette(alone_job.previews[1], alone.palettes[1], "new sprite")

-- Walk sheet: 16 frames tagged by direction and pose, in the active sprite
-- or in a new one. klein base takes about a minute per variant.
local walk_variants = backend == "comfyui" and 1 or variants
local walk_options = {
  bridge_command = app.params.bridge,
  backend = backend,
  task = "walk-sheet",
  prompt = "a knight with a blue cape",
  use_palette = true,
  variants = walk_variants,
}
local walk_names = { "down", "left", "right", "up", "arms raised", "jump left", "jump right",
                     "lying" }

local function assert_walk_tags(sprite, first)
  local tags = {}
  for _, tag in ipairs(sprite.tags) do
    tags[tag.name] = tag
  end
  for _, name in ipairs(walk_names) do
    assert(tags[name], "missing tag " .. name)
  end
  assert(tags.down.fromFrame.frameNumber == first and tags.down.toFrame.frameNumber == first + 2,
         "the down tag does not hold the first 3 new frames")
  assert(tags.up.aniDir == AniDir.PING_PONG, "the walk does not loop back and forth")
  assert(tags.lying.fromFrame.frameNumber == first + 15, "the last pose is not last")
end

local walk_sprite = character_sprite
local frames_before = #walk_sprite.frames
local walk_job = assert(inpaint.generate(walk_sprite, walk_sprite.frames[1], walk_options))
assert(walk_job.walk and walk_job.region == Rectangle(0, 0, 128, 128), "unexpected walk region")
assert(#walk_job.patches == walk_variants, "the walk run returned the wrong variant count")
for i, preview in ipairs(walk_job.previews) do
  assert_in_palette(preview, walk_sprite.palettes[1], "walk sheet " .. i)
end
inpaint.apply_walk(walk_sprite, walk_job, 1)
assert(#walk_sprite.frames == frames_before + 16, "expected 16 new frames")
assert_walk_tags(walk_sprite, frames_before + 1)
local walk_layer = walk_sprite.layers[#walk_sprite.layers]
assert(walk_layer.name == "AI walk", "unexpected layer: " .. walk_layer.name)
local first_cel = walk_layer:cel(frames_before + 1)
assert(first_cel and first_cel.image.colorMode == walk_sprite.colorMode,
       "walk frame color mode differs")
if backend == "invert" then
  assert(first_cel.position == Point(-12, -12), "the walk frame is not centered: "
         .. tostring(first_cel.bounds))
end

-- Edit mode: the active sprite is the character to animate, no description.
local reference_options = {}
for key, value in pairs(walk_options) do
  reference_options[key] = value
end
reference_options.from_sprite = true
reference_options.prompt = ""
local reference_run = assert(inpaint.start(walk_sprite, walk_sprite.frames[1], reference_options))
assert(reference_run.paths.reference and app.fs.isFile(reference_run.paths.reference),
       "the reference sprite was not written")
assert(reference_run.command:find("--reference", 1, true), "the reference is not sent")
local reference_finished, reference_job = wait_run(reference_run, 600)
assert(reference_finished, reference_job)
assert(not app.fs.isFile(reference_run.paths.reference), "the reference file was left")
assert(#reference_job.previews == walk_variants, "the edit run returned the wrong variant count")
assert_in_palette(reference_job.previews[1], walk_sprite.palettes[1], "edited walk sheet")

walk_options.new_sprite = true
local walk_run = assert(inpaint.start(nil, nil, walk_options))
local walk_finished, new_walk_job = wait_run(walk_run, 600)
assert(walk_finished, new_walk_job)
local walk = inpaint.apply_walk(nil, new_walk_job, nil)
assert(walk.width == 32 and walk.height == 32, "the walk sprite has the wrong size")
assert(#walk.frames == 16, "expected 16 frames")
assert_walk_tags(walk, 1)
assert(#walk.layers == 1 and walk.layers[1].isGroup, "expected one group of variants")
assert(#walk.layers[1].layers == walk_variants, "expected one layer per variant")

print("ROUNDTRIP OK (" .. mode .. ", " .. backend .. ", " .. (app.params.model or "default model")
      .. ", " .. variants .. " variants)")
