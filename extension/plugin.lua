local DEFAULT_PREFERENCES = {
  bridge_command = "ai-inpaint-bridge",
  margin = 8,
  backend = "comfyui",
  prompt = "",
  denoise = 80,
  use_palette = true,
  variants = 4,
  output = "New layer",
}
local OUTPUTS = { "New layer", "Active layer" }
local BACKENDS = { "comfyui", "invert" }
local MAX_ALERT_LINES = 15
local MAX_VARIANTS = 8
local PREVIEW_COLUMNS = 4
local PREVIEW_CELL = 160
local PREVIEW_GAP = 8
local CHECKER_SIZE = 8
local CHECKER_LIGHT = Color{ r = 204, g = 204, b = 204, a = 255 }
local CHECKER_DARK = Color{ r = 153, g = 153, b = 153, a = 255 }
local SELECTED_COLOR = Color{ r = 255, g = 196, b = 0, a = 255 }
local UNCHANGED_COLOR = Color{ r = 220, g = 40, b = 40, a = 255 }
-- Below this share of changed selected pixels a variant is shown as a miss.
local UNCHANGED_THRESHOLD = 0.05

local inpaint

local function preferences(plugin)
  for key, value in pairs(DEFAULT_PREFERENCES) do
    if plugin.preferences[key] == nil then
      plugin.preferences[key] = value
    end
  end
  return plugin.preferences
end

local function alert_lines(message)
  local lines = {}
  for line in message:gmatch("[^\n]+") do
    if #lines == MAX_ALERT_LINES then
      lines[#lines + 1] = "..."
      break
    end
    lines[#lines + 1] = line
  end
  return lines
end

local function show_settings(plugin)
  local prefs = preferences(plugin)
  local dlg = Dialog("AI Settings")
  dlg:combobox{ id = "backend", label = "Backend", option = prefs.backend, options = BACKENDS }
  dlg:entry{ id = "bridge_command", label = "Bridge command", text = prefs.bridge_command }
  dlg:number{ id = "margin", label = "Context margin (px)",
              text = tostring(prefs.margin), decimals = 0 }
  dlg:button{ id = "ok", text = "OK", focus = true }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if dlg.data.ok then
    prefs.backend = dlg.data.backend
    prefs.bridge_command = dlg.data.bridge_command
    prefs.margin = math.max(0, math.floor(dlg.data.margin))
  end
end

-- Returns false when the user cancels.
local function ask_inpaint(prefs)
  local dlg = Dialog("AI Inpaint")
  dlg:entry{ id = "prompt", label = "Prompt", text = prefs.prompt, focus = true }
  dlg:slider{ id = "denoise", label = "Change strength (%)", min = 0, max = 100,
              value = prefs.denoise }
  dlg:check{ id = "use_palette", text = "Use the sprite palette", selected = prefs.use_palette }
  dlg:slider{ id = "variants", label = "Variants", min = 1, max = MAX_VARIANTS,
              value = prefs.variants }
  dlg:combobox{ id = "output", label = "Output", option = prefs.output, options = OUTPUTS }
  dlg:button{ id = "ok", text = "Generate" }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if not dlg.data.ok then
    return false
  end
  prefs.prompt = dlg.data.prompt
  prefs.denoise = dlg.data.denoise
  prefs.use_palette = dlg.data.use_palette
  prefs.variants = dlg.data.variants
  prefs.output = dlg.data.output
  return true
end

local function run_options(prefs)
  return {
    bridge_command = prefs.bridge_command,
    margin = prefs.margin,
    backend = prefs.backend,
    prompt = prefs.prompt,
    denoise = prefs.denoise / 100,
    use_palette = prefs.use_palette,
    variants = prefs.backend == "comfyui" and prefs.variants or 1,
  }
end

local function preview_layout(job)
  local region = job.region
  -- Integer zoom keeps pixels square; regions larger than a cell are shrunk.
  local scale = PREVIEW_CELL / math.max(region.width, region.height)
  if scale >= 1 then
    scale = math.floor(scale)
  end
  local count = #job.previews
  local columns = math.min(count, PREVIEW_COLUMNS)
  local rows = math.ceil(count / columns)
  local cell_w = math.max(1, math.floor(region.width * scale))
  local cell_h = math.max(1, math.floor(region.height * scale))
  return {
    columns = columns,
    cell_w = cell_w,
    cell_h = cell_h,
    width = columns * cell_w + (columns + 1) * PREVIEW_GAP,
    height = rows * cell_h + (rows + 1) * PREVIEW_GAP,
  }
end

local function cell_rect(layout, index)
  local column = (index - 1) % layout.columns
  local row = (index - 1) // layout.columns
  return Rectangle(PREVIEW_GAP + column * (layout.cell_w + PREVIEW_GAP),
                   PREVIEW_GAP + row * (layout.cell_h + PREVIEW_GAP),
                   layout.cell_w, layout.cell_h)
end

local function paint_checker(ctx, rect)
  for y = rect.y, rect.y + rect.height - 1, CHECKER_SIZE do
    for x = rect.x, rect.x + rect.width - 1, CHECKER_SIZE do
      local light = ((x - rect.x) // CHECKER_SIZE + (y - rect.y) // CHECKER_SIZE) % 2 == 0
      ctx.color = light and CHECKER_LIGHT or CHECKER_DARK
      ctx:fillRect(Rectangle(x, y, math.min(CHECKER_SIZE, rect.x + rect.width - x),
                             math.min(CHECKER_SIZE, rect.y + rect.height - y)))
    end
  end
end

-- Returns "one" and the chosen index, "all", or nil when the user cancels.
local function pick_variant(job)
  local layout = preview_layout(job)
  local unchanged = {}
  local selected
  for i = 1, #job.previews do
    unchanged[i] = inpaint.changed_fraction(job, i) < UNCHANGED_THRESHOLD
    if not selected and not unchanged[i] then
      selected = i
    end
  end
  selected = selected or 1
  local dlg = Dialog("AI - Pick a variant")
  local function update_label()
    local text = "Variant " .. selected .. " of " .. #job.previews
    if unchanged[selected] then
      text = text .. " (nearly unchanged)"
    end
    dlg:modify{ id = "current", text = text }
  end
  dlg:canvas{
    id = "previews",
    width = layout.width,
    height = layout.height,
    onpaint = function(ev)
      local ctx = ev.context
      ctx.antialias = false
      for i, preview in ipairs(job.previews) do
        local rect = cell_rect(layout, i)
        paint_checker(ctx, rect)
        ctx:drawImage(preview, Rectangle(0, 0, preview.width, preview.height), rect)
        if unchanged[i] then
          ctx.color = UNCHANGED_COLOR
          ctx.strokeWidth = 2
          ctx:beginPath()
          ctx:moveTo(rect.x, rect.y)
          ctx:lineTo(rect.x + rect.width, rect.y + rect.height)
          ctx:moveTo(rect.x + rect.width, rect.y)
          ctx:lineTo(rect.x, rect.y + rect.height)
          ctx:stroke()
        end
        if i == selected then
          ctx.color = SELECTED_COLOR
          ctx.strokeWidth = 3
          ctx:strokeRect(Rectangle(rect.x - 2, rect.y - 2, rect.width + 4, rect.height + 4))
        end
      end
    end,
    onmousedown = function(ev)
      for i = 1, #job.previews do
        if cell_rect(layout, i):contains(Point(ev.x, ev.y)) then
          selected = i
          update_label()
          dlg:repaint()
        end
      end
    end,
  }
  dlg:label{ id = "current", text = "" }
  dlg:button{ id = "apply", text = "Apply", focus = true }
  dlg:button{ id = "all", text = "Keep all as layers" }
  dlg:button{ id = "cancel", text = "Cancel" }
  update_label()
  dlg:show()
  if dlg.data.apply then
    return "one", selected
  end
  if dlg.data.all then
    return "all"
  end
  return nil
end

local function apply_one(sprite, frame, job, index, target)
  if target then
    inpaint.apply_active_layer(sprite, frame, job, index, target)
  else
    inpaint.apply_new_layer(sprite, frame, job, index)
  end
end

-- target is the layer to write into, or nil for a new layer.
local function apply_job(sprite, frame, job, target)
  if #job.patches == 1 then
    apply_one(sprite, frame, job, 1, target)
    return
  end
  local choice, index = pick_variant(job)
  if choice == "one" then
    apply_one(sprite, frame, job, index, target)
  elseif choice == "all" then
    inpaint.apply_all_layers(sprite, frame, job)
  end
end

-- One entry per bridge task: ask fills the preferences from the task's dialog
-- and returns false when the user cancels, generate returns a job or nil and a
-- message.
local function task_entries()
  return {
    {
      name = "inpaint",
      command_id = "AiInpaintSelection",
      title = "Inpaint Selection",
      enabled = function()
        return app.sprite ~= nil and not app.sprite.selection.isEmpty
      end,
      ask = ask_inpaint,
      generate = inpaint.generate,
    },
  }
end

local function run_task(plugin, task)
  local prefs = preferences(plugin)
  if prefs.backend == "comfyui" and not task.ask(prefs) then
    return
  end
  local sprite, frame = app.sprite, app.frame
  local target
  if prefs.output == "Active layer" then
    local problem = inpaint.active_layer_problem(app.layer)
    if problem then
      app.alert{ title = "AI", text = problem }
      return
    end
    target = app.layer
  end
  local options = run_options(prefs)
  options.task = task.name
  local job, message = task.generate(sprite, frame, options)
  if not job then
    app.alert{ title = "AI", text = alert_lines(message) }
    return
  end
  apply_job(sprite, frame, job, target)
end

function init(plugin)
  inpaint = dofile(app.fs.joinPath(plugin.path, "inpaint.lua"))

  plugin:newMenuGroup{ id = "ai_menu", title = "AI", group = "edit_new" }

  for _, task in ipairs(task_entries()) do
    plugin:newCommand{
      id = task.command_id,
      title = task.title,
      group = "ai_menu",
      onenabled = task.enabled,
      onclick = function()
        run_task(plugin, task)
      end,
    }
  end

  plugin:newMenuSeparator{ group = "ai_menu" }

  plugin:newCommand{
    id = "AiInpaintSettings",
    title = "Settings...",
    group = "ai_menu",
    onclick = function()
      show_settings(plugin)
    end,
  }
end
