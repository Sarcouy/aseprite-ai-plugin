local inpaint = {}

local LAYER_NAME = "AI inpaint"
local CHARACTER_LAYER_NAME = "AI character"
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

-- sprite is nil for a result that goes to a new RGB sprite.
local function build_patch(sprite, result, mask)
  local mode = sprite and sprite.colorMode or ColorMode.RGB
  local patch = Image(mask.width, mask.height, mode)
  local to_index
  if mode == ColorMode.INDEXED then
    patch:clear(sprite.transparentColor)
    to_index = index_finder(sprite)
  end
  for pixel in result:pixels() do
    if mask:getPixel(pixel.x, pixel.y) == WHITE then
      local value = pixel()
      patch:putPixel(pixel.x, pixel.y, to_index and to_index(value) or value)
    end
  end
  return patch
end

local function bridge_command(options, paths)
  local cmd = string.format(
    "%s --task %s --output %s --backend %s --prompt %s --variants %d",
    options.bridge_command, shell_quote(options.task or "inpaint"),
    shell_quote(paths.output), shell_quote(options.backend or "invert"),
    shell_quote(options.prompt or ""), #paths.variants)
  if paths.input then
    -- LC_NUMERIC may use a decimal comma, which argparse would reject.
    local denoise = string.format("%.2f", options.denoise or 0.8):gsub(",", ".")
    cmd = cmd .. " --input " .. shell_quote(paths.input) .. " --mask "
          .. shell_quote(paths.mask) .. " --denoise " .. denoise
  end
  if options.size then
    cmd = cmd .. string.format(" --size %dx%d", options.size.width, options.size.height)
  end
  if options.model then
    cmd = cmd .. " --model " .. shell_quote(options.model)
  end
  if paths.palette then
    cmd = cmd .. " --palette " .. shell_quote(paths.palette)
  end
  return cmd
end

local function read_file(path)
  local file = io.open(path, "r")
  if not file then
    return nil
  end
  local text = file:read("a")
  file:close()
  return text
end

-- The bridge writes variant i of "<run>-output.png" to "<run>-output-<i>.png".
-- In background runs it also gets a log, its process id and a status file
-- holding its exit code, written last. Tasks drawing from the prompt alone
-- have no input region nor mask.
local function temp_paths(with_input, with_palette, variants)
  local dir = app.fs.joinPath(app.fs.tempPath, "aseprite-ai-inpaint")
  app.fs.makeAllDirectories(dir)
  local run_id = os.time() .. "-" .. math.random(100000, 999999)
  local function path(name, extension)
    return app.fs.joinPath(dir, run_id .. "-" .. name .. (extension or ".png"))
  end
  local outputs = {}
  for i = 1, variants do
    outputs[i] = path("output-" .. i)
  end
  return {
    input = with_input and path("input") or nil,
    mask = with_input and path("mask") or nil,
    output = path("output"),
    palette = with_palette and path("palette") or nil,
    variants = outputs,
    log = path("log", ".txt"),
    pid = path("pid", ".txt"),
    status = path("status", ".txt"),
  }
end

local function remove_files(paths)
  for _, key in ipairs({ "input", "mask", "palette", "log", "pid", "status" }) do
    if paths[key] then
      os.remove(paths[key])
    end
  end
  os.remove(paths.status .. ".tmp")
  for _, path in ipairs(paths.variants) do
    os.remove(path)
  end
end

local function read_variants(paths, region, output)
  local results = {}
  for i, path in ipairs(paths.variants) do
    if not app.fs.isFile(path) then
      return nil, "The bridge produced no image for variant " .. i .. ":\n" .. output
    end
    local result = Image{ fromFile = path }
    if not result or result.width ~= region.width or result.height ~= region.height then
      return nil, "The bridge returned an image of unexpected size."
    end
    results[i] = result
  end
  return results
end

-- Rectangle of the drawn character: centered on the selection, or on the
-- canvas without one, at the origin of a new sprite.
local function character_region(sprite, size)
  if not sprite then
    return Rectangle(0, 0, size.width, size.height)
  end
  local area = sprite.selection.isEmpty and sprite.bounds or sprite.selection.bounds
  return Rectangle(area.x + (area.width - size.width) // 2,
                   area.y + (area.height - size.height) // 2, size.width, size.height)
end

-- The character task draws a whole region: its mask is fully selected and its
-- original fully transparent, so that an empty variant shows as unchanged. A
-- result going to a new sprite is not bound to the active sprite, which only
-- lends its palette.
local function prepare_character(sprite, options)
  local placed = sprite and not options.new_sprite and sprite or nil
  local region = character_region(placed, options.size)
  local use_palette = sprite and (options.use_palette or sprite.colorMode == ColorMode.INDEXED)
  local paths = temp_paths(false, use_palette, options.variants or 1)
  local mask = Image(region.width, region.height, ColorMode.RGB)
  mask:clear(WHITE)
  local palette
  if use_palette then
    export_palette(sprite, paths.palette)
    palette = Palette(sprite.palettes[1])
  end
  return { sprite = placed, region = region, mask = mask,
           original = Image(region.width, region.height, ColorMode.RGB), paths = paths,
           palette = palette, layer_name = CHARACTER_LAYER_NAME,
           command = bridge_command(options, paths) }
end

-- Writes the region, its mask and the palette, and returns a run holding what
-- the result needs, or nil and a message for the user.
local function prepare(sprite, frame, options)
  if sprite and sprite.colorMode == ColorMode.GRAY then
    return nil, "Grayscale sprites are not supported."
  end
  if options.task == "character" then
    return prepare_character(sprite, options)
  end
  local selection = sprite.selection
  if selection.isEmpty then
    return nil, "Select the area to rewrite first."
  end
  local region = context_region(sprite, selection, options.margin)
  if region.isEmpty then
    return nil, "The selection is outside the canvas."
  end

  local use_palette = options.use_palette or sprite.colorMode == ColorMode.INDEXED
  local paths = temp_paths(true, use_palette, options.variants or 1)
  local mask = build_mask(selection, region)
  local original = render_region(sprite, frame, region)
  original:saveAs(paths.input)
  mask:saveAs(paths.mask)
  if use_palette then
    export_palette(sprite, paths.palette)
  end
  return { sprite = sprite, region = region, mask = mask, original = original, paths = paths,
           layer_name = LAYER_NAME, command = bridge_command(options, paths) }
end

-- Reads the variants and removes the run's files. Returns a job { region,
-- mask, original, previews, patches, layer_name, palette } on success, or nil
-- and a message for the user. previews are RGBA images of the whole region,
-- patches hold only the selected pixels in the sprite color mode (RGB without
-- a sprite); palette is the one to give a new sprite, if any.
local function finish(run, ok, output)
  local results, message
  if ok then
    results, message = read_variants(run.paths, run.region, output)
  else
    message = "The bridge failed:\n" .. output
  end
  remove_files(run.paths)
  if not results then
    return nil, message
  end
  local patches = {}
  for i, result in ipairs(results) do
    patches[i] = build_patch(run.sprite, result, run.mask)
  end
  return { region = run.region, mask = run.mask, original = run.original, previews = results,
           patches = patches, layer_name = run.layer_name, palette = run.palette }
end

-- Runs the bridge and waits for it: Aseprite is frozen meanwhile. Used by the
-- batch tests, where no timer can tick.
function inpaint.generate(sprite, frame, options)
  local run, message = prepare(sprite, frame, options)
  if not run then
    return nil, message
  end
  local handle = io.popen(run.command .. " 2>&1")
  if not handle then
    remove_files(run.paths)
    return nil, "Unable to start: " .. options.bridge_command
  end
  local output = handle:read("a")
  return finish(run, handle:close() == true, output)
end

-- Starts the bridge in the background and returns the run to pass to poll and
-- cancel, or nil and a message for the user.
function inpaint.start(sprite, frame, options)
  local run, message = prepare(sprite, frame, options)
  if not run then
    return nil, message
  end
  local paths = run.paths
  local script = string.format(
    "%s > %s 2>&1 & echo $! > %s; wait $!; echo $? > %s; mv %s %s",
    run.command, shell_quote(paths.log), shell_quote(paths.pid),
    shell_quote(paths.status .. ".tmp"), shell_quote(paths.status .. ".tmp"),
    shell_quote(paths.status))
  if not os.execute("sh -c " .. shell_quote(script) .. " > /dev/null 2>&1 &") then
    remove_files(paths)
    return nil, "Unable to start: " .. options.bridge_command
  end
  return run
end

-- Closing a sprite leaves a dangling object that raises on any access.
function inpaint.is_closed(sprite)
  return not pcall(function() return sprite.colorMode end)
end

local function kill(run)
  local pid = tonumber(read_file(run.paths.pid) or "")
  if pid then
    os.execute("kill " .. pid .. " 2> /dev/null")
    run.killed = true
  end
end

-- Returns nil while the bridge runs, then true and the job, or false and a
-- message for the user. A cancelled run ends with false and no message; a run
-- whose sprite was closed is cancelled.
function inpaint.poll(run)
  local closed = not run.cancelled and run.sprite ~= nil and inpaint.is_closed(run.sprite)
  if closed then
    inpaint.cancel(run)
    run.closed_message = "The sprite was closed during the generation."
  end
  if run.cancelled and not run.killed then
    kill(run)
  end
  local status = read_file(run.paths.status)
  if not status then
    return nil
  end
  if run.cancelled then
    remove_files(run.paths)
    return false, run.closed_message
  end
  local output = read_file(run.paths.log) or ""
  local job, message = finish(run, tonumber(status) == 0, output)
  return job ~= nil, job or message
end

-- Stops the bridge; poll then reports the end of the run.
function inpaint.cancel(run)
  run.cancelled = true
  kill(run)
end

-- Share of the selected pixels that the variant changed, from 0 to 1. Two
-- fully transparent pixels are equal whatever their hidden color.
function inpaint.changed_fraction(job, index)
  local preview, selected, changed = job.previews[index], 0, 0
  for y = 0, job.region.height - 1 do
    for x = 0, job.region.width - 1 do
      if job.mask:getPixel(x, y) == WHITE then
        selected = selected + 1
        local before, after = job.original:getPixel(x, y), preview:getPixel(x, y)
        if before ~= after and (pc.rgbaA(before) > 0 or pc.rgbaA(after) > 0) then
          changed = changed + 1
        end
      end
    end
  end
  return selected > 0 and changed / selected or 0
end

-- Adds the chosen patch of a job in a new layer, as a single undo step.
function inpaint.apply_new_layer(sprite, frame, job, index)
  local name = job.layer_name or LAYER_NAME
  app.transaction(name, function()
    local layer = sprite:newLayer()
    layer.name = name
    sprite:newCel(layer, frame, job.patches[index], Point(job.region.x, job.region.y))
  end)
end

-- Adds every patch of a job in a group, one layer per variant, only the first
-- one visible, as a single undo step.
function inpaint.apply_all_layers(sprite, frame, job)
  local name = job.layer_name or LAYER_NAME
  app.transaction(name, function()
    local group = sprite:newGroup()
    group.name = name
    for i = #job.patches, 1, -1 do
      local layer = sprite:newLayer()
      layer.name = "Variant " .. i
      layer.parent = group
      layer.isVisible = i == 1
      sprite:newCel(layer, frame, job.patches[i], Point(job.region.x, job.region.y))
    end
  end)
end

-- Palette of a sprite without source palette: transparent first, then the
-- colors of the previews in order of appearance.
local function previews_palette(previews)
  local colors, seen = {}, {}
  for _, preview in ipairs(previews) do
    for pixel in preview:pixels() do
      local value = pixel()
      if pc.rgbaA(value) > 0 and not seen[value] then
        seen[value] = true
        colors[#colors + 1] = value
      end
    end
  end
  local palette = Palette(#colors + 1)
  palette:setColor(0, Color{ r = 0, g = 0, b = 0, a = 0 })
  for i, value in ipairs(colors) do
    palette:setColor(i, Color(value))
  end
  return palette
end

-- Creates an RGB sprite of the job's size and puts the chosen variant in it,
-- or every variant when index is nil. Returns the sprite.
function inpaint.apply_new_sprite(job, index)
  local region = job.region
  local sprite = Sprite(region.width, region.height, ColorMode.RGB)
  sprite:setPalette(job.palette or previews_palette(job.previews))
  local first = sprite.layers[1]
  local placed = { region = Rectangle(0, 0, region.width, region.height), mask = job.mask,
                   original = job.original, previews = job.previews, patches = job.previews,
                   layer_name = job.layer_name }
  if index then
    inpaint.apply_new_layer(sprite, sprite.frames[1], placed, index)
  else
    inpaint.apply_all_layers(sprite, sprite.frames[1], placed)
  end
  sprite:deleteLayer(first)
  return sprite
end

-- Returns nil when the layer can receive a patch, otherwise a message for the user.
function inpaint.active_layer_problem(layer)
  if not layer then
    return "Select a layer first."
  end
  if layer.isGroup or layer.isTilemap or layer.isReference then
    return "The active layer must be a regular image layer."
  end
  if not layer.isEditable then
    return "The active layer is locked."
  end
  return nil
end

-- Writes the chosen patch into the layer, selected pixels included when they
-- are transparent, so that the result can erase the original. On a background
-- layer transparency becomes the background color, as with the eraser.
function inpaint.apply_active_layer(sprite, frame, job, index, layer)
  local patch, region = job.patches[index], job.region
  local transparent = sprite.colorMode == ColorMode.INDEXED and sprite.transparentColor or 0
  local erased = transparent
  if layer.isBackground then
    erased = sprite.colorMode == ColorMode.INDEXED and app.bgColor.index or app.bgColor.rgbaPixel
  end
  app.transaction(LAYER_NAME, function()
    local cel = layer:cel(frame)
    local canvas = Image(sprite.spec)
    canvas:clear(transparent)
    if cel then
      canvas:drawImage(cel.image, cel.position)
    end
    for y = 0, region.height - 1 do
      for x = 0, region.width - 1 do
        if job.mask:getPixel(x, y) == WHITE then
          local value = patch:getPixel(x, y)
          local is_transparent = value == transparent
          if sprite.colorMode ~= ColorMode.INDEXED then
            is_transparent = pc.rgbaA(value) == 0
          end
          canvas:putPixel(region.x + x, region.y + y, is_transparent and erased or value)
        end
      end
    end
    if cel then
      cel.image = canvas
      cel.position = Point(0, 0)
    else
      sprite:newCel(layer, frame, canvas, Point(0, 0))
    end
  end)
end

return inpaint
