-- Batch test: aseprite -b --script-param bridge=<command> --script tests/roundtrip.lua

local script_dir = app.fs.filePath(debug.getinfo(1, "S").source:sub(2))
local inpaint = dofile(app.fs.joinPath(script_dir, "..", "extension", "inpaint.lua"))

local RED = app.pixelColor.rgba(200, 10, 30, 255)
local INVERTED_RED = app.pixelColor.rgba(55, 245, 225, 255)

local sprite = Sprite(8, 8, ColorMode.RGB)
local background = Image(8, 8, ColorMode.RGB)
background:clear(RED)
sprite:newCel(sprite.layers[1], 1, background, Point(0, 0))
sprite.selection = Selection(Rectangle(2, 2, 3, 3))

local backend = app.params.backend or "invert"
local ok, message = inpaint.run(sprite, sprite.frames[1], {
  bridge_command = app.params.bridge,
  margin = 2,
  backend = backend,
  prompt = "a blue gem",
  denoise = 1.0,
})
assert(ok, message)

local top = sprite.layers[#sprite.layers]
assert(top.name == "AI inpaint", "unexpected top layer: " .. top.name)

local flat = Image(8, 8, ColorMode.RGB)
flat:drawSprite(sprite, 1)
if backend == "invert" then
  assert(flat:getPixel(3, 3) == INVERTED_RED, "selected pixel was not rewritten")
  assert(flat:getPixel(4, 4) == INVERTED_RED, "selection corner was not rewritten")
end
assert(flat:getPixel(1, 1) == RED, "pixel outside the selection was modified")
assert(flat:getPixel(5, 5) == RED, "pixel outside the selection was modified")

print("ROUNDTRIP OK")
