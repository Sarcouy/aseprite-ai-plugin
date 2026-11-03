local inpaint = {}

local LAYER_NAME = "AI inpaint"
local WHITE = app.pixelColor.rgba(255, 255, 255, 255)

local function shell_quote(text)
  return "'" .. text:gsub("'", "'\\''") .. "'"
end

local function context_region(sprite, selection, margin)
  local bounds = selection.bounds
  local region = Rectangle(bounds.x - margin, bounds.y - margin,
                           bounds.width + 2 * margin, bounds.height + 2 * margin)
  return region:intersect(sprite.bounds)
end

local function render_region(sprite, frame, region)
  local flat = Image(sprite.width, sprite.height, ColorMode.RGB)
  flat:drawSprite(sprite, frame)
  return Image(flat, region)
end

local function build_mask(selection, region)
  local mask = Image(region.width, region.height, ColorMode.RGB)
  for y = 0, region.height - 1 do
    for x = 0, region.width - 1 do
      if selection:contains(region.x + x, region.y + y) then
        mask:putPixel(x, y, WHITE)
      end
    end
  end
  return mask
end

local function build_patch(result, selection, region)
  local patch = Image(region.width, region.height, ColorMode.RGB)
  for pixel in result:pixels() do
    if selection:contains(region.x + pixel.x, region.y + pixel.y) then
      patch:putPixel(pixel.x, pixel.y, pixel())
    end
  end
  return patch
end

local function run_bridge(options, paths)
  -- LC_NUMERIC may use a decimal comma, which argparse would reject.
  local denoise = string.format("%.2f", options.denoise or 0.8):gsub(",", ".")
  local cmd = string.format(
    "%s --input %s --mask %s --output %s --backend %s --prompt %s --denoise %s 2>&1",
    options.bridge_command, shell_quote(paths.input), shell_quote(paths.mask),
    shell_quote(paths.output), shell_quote(options.backend or "invert"),
    shell_quote(options.prompt or ""), denoise)
  local handle = io.popen(cmd)
  if not handle then
    return false, "Unable to start: " .. options.bridge_command
  end
  local output = handle:read("a")
  return handle:close() == true, output
end

local function temp_paths()
  local dir = app.fs.joinPath(app.fs.tempPath, "aseprite-ai-inpaint")
  app.fs.makeAllDirectories(dir)
  local run_id = os.time() .. "-" .. math.random(100000, 999999)
  local function path(name)
    return app.fs.joinPath(dir, run_id .. "-" .. name .. ".png")
  end
  return { input = path("input"), mask = path("mask"), output = path("output") }
end

-- Returns true on success, or false and a message for the user.
function inpaint.run(sprite, frame, options)
  if sprite.colorMode ~= ColorMode.RGB then
    return false, "Only RGB sprites are supported for now."
  end
  local selection = sprite.selection
  if selection.isEmpty then
    return false, "Select the area to rewrite first."
  end
  local region = context_region(sprite, selection, options.margin)
  if region.isEmpty then
    return false, "The selection is outside the canvas."
  end

  local paths = temp_paths()
  render_region(sprite, frame, region):saveAs(paths.input)
  build_mask(selection, region):saveAs(paths.mask)

  local ok, output = run_bridge(options, paths)
  if not ok then
    return false, "The bridge failed:\n" .. output
  end
  if not app.fs.isFile(paths.output) then
    return false, "The bridge produced no image:\n" .. output
  end
  local result = Image{ fromFile = paths.output }
  if not result or result.width ~= region.width or result.height ~= region.height then
    return false, "The bridge returned an image of unexpected size."
  end

  local patch = build_patch(result, selection, region)
  app.transaction(LAYER_NAME, function()
    local layer = sprite:newLayer()
    layer.name = LAYER_NAME
    sprite:newCel(layer, frame, patch, Point(region.x, region.y))
  end)
  return true
end

return inpaint
