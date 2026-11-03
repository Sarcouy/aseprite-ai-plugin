-- Batch test: aseprite -b --script-param bridge=<command> [--script-param backend=comfyui]
--             [--script-param mode=indexed] --script tests/roundtrip.lua

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
local ok, message = inpaint.run(sprite, sprite.frames[1], {
  bridge_command = app.params.bridge,
  margin = 2,
  backend = backend,
  prompt = "a blue gem",
  denoise = 1.0,
  use_palette = true,
})
assert(ok, message)

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

print("ROUNDTRIP OK (" .. mode .. ", " .. backend .. ")")
