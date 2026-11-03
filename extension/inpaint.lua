local inpaint = {}

local LAYER_NAME = "AI inpaint"
local WHITE = app.pixelColor.rgba(255, 255, 255, 255)
local pc = app.pixelColor

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

local function export_palette(sprite, path)
  local palette = sprite.palettes[1]
  local strip = Image(#palette, 1, ColorMode.RGB)
  for i = 0, #palette - 1 do
    strip:putPixel(i, 0, palette:getColor(i).rgbaPixel)
  end
  strip:saveAs(path)
end

-- Maps an RGBA pixel to its palette index, falling back to the nearest color.
local function index_finder(sprite)
  local palette = sprite.palettes[1]
  local exact, entries = {}, {}
  for i = #palette - 1, 0, -1 do
    local color = palette:getColor(i)
    if color.alpha > 0 and i ~= sprite.transparentColor then
      exact[pc.rgba(color.red, color.green, color.blue, 255)] = i
      entries[#entries + 1] = { index = i, r = color.red, g = color.green, b = color.blue }
    end
  end
  return function(value)
    if pc.rgbaA(value) == 0 then
      return sprite.transparentColor
    end
    local found = exact[pc.rgba(pc.rgbaR(value), pc.rgbaG(value), pc.rgbaB(value), 255)]
    if found then
      return found
    end
    local best, best_distance = sprite.transparentColor, math.huge
    for _, entry in ipairs(entries) do
      local distance = (entry.r - pc.rgbaR(value)) ^ 2 + (entry.g - pc.rgbaG(value)) ^ 2
                       + (entry.b - pc.rgbaB(value)) ^ 2
      if distance < best_distance then
        best, best_distance = entry.index, distance
      end
    end
    return best
  end
end

local function build_patch(sprite, result, selection, region)
  local patch = Image(region.width, region.height, sprite.colorMode)
  local to_index
  if sprite.colorMode == ColorMode.INDEXED then
    patch:clear(sprite.transparentColor)
    to_index = index_finder(sprite)
  end
  for pixel in result:pixels() do
    if selection:contains(region.x + pixel.x, region.y + pixel.y) then
      local value = pixel()
      patch:putPixel(pixel.x, pixel.y, to_index and to_index(value) or value)
    end
  end
  return patch
end

local function run_bridge(options, paths)
  -- LC_NUMERIC may use a decimal comma, which argparse would reject.
  local denoise = string.format("%.2f", options.denoise or 0.8):gsub(",", ".")
  local cmd = string.format(
    "%s --input %s --mask %s --output %s --backend %s --prompt %s --denoise %s",
    options.bridge_command, shell_quote(paths.input), shell_quote(paths.mask),
    shell_quote(paths.output), shell_quote(options.backend or "invert"),
    shell_quote(options.prompt or ""), denoise)
  if paths.palette then
    cmd = cmd .. " --palette " .. shell_quote(paths.palette)
  end
  local handle = io.popen(cmd .. " 2>&1")
  if not handle then
    return false, "Unable to start: " .. options.bridge_command
  end
  local output = handle:read("a")
  return handle:close() == true, output
end

local function temp_paths(with_palette)
  local dir = app.fs.joinPath(app.fs.tempPath, "aseprite-ai-inpaint")
  app.fs.makeAllDirectories(dir)
  local run_id = os.time() .. "-" .. math.random(100000, 999999)
  local function path(name)
    return app.fs.joinPath(dir, run_id .. "-" .. name .. ".png")
  end
  return {
    input = path("input"),
    mask = path("mask"),
    output = path("output"),
    palette = with_palette and path("palette") or nil,
  }
end

-- Returns true on success, or false and a message for the user.
function inpaint.run(sprite, frame, options)
  if sprite.colorMode == ColorMode.GRAY then
    return false, "Grayscale sprites are not supported."
  end
  local selection = sprite.selection
  if selection.isEmpty then
    return false, "Select the area to rewrite first."
  end
  local region = context_region(sprite, selection, options.margin)
  if region.isEmpty then
    return false, "The selection is outside the canvas."
  end

  local use_palette = options.use_palette or sprite.colorMode == ColorMode.INDEXED
  local paths = temp_paths(use_palette)
  render_region(sprite, frame, region):saveAs(paths.input)
  build_mask(selection, region):saveAs(paths.mask)
  if use_palette then
    export_palette(sprite, paths.palette)
  end

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

  local patch = build_patch(sprite, result, selection, region)
  app.transaction(LAYER_NAME, function()
    local layer = sprite:newLayer()
    layer.name = LAYER_NAME
    sprite:newCel(layer, frame, patch, Point(region.x, region.y))
  end)
  return true
end

return inpaint
