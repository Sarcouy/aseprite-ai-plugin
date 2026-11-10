local DEFAULT_PREFERENCES = {
  bridge_command = "ai-inpaint-bridge",
  margin = 8,
  backend = "comfyui",
  inpaint_model = "klein-4b",
  prompt = "",
  denoise = 80,
  use_palette = true,
  variants = 4,
  output = "New layer",
  character_prompt = "",
  character_size = "32",
  character_output = "New layer",
  walk_prompt = "",
  walk_output = "New sprite",
  walk_from_sprite = false,
  walk_variants = 2,
  animation = "idle",
  animation_prompt = "",
  animation_variants = 2,
  rotate_prompt = "",
  rotate_variants = 2,
}
local OUTPUTS = { "New layer", "Active layer" }
local CHARACTER_OUTPUTS = { "New layer", "New sprite" }
local CHARACTER_SIZES = { "16", "24", "32", "48", "64" }
local WALK_OUTPUTS = { "New sprite", "New frames" }
local BACKENDS = { "comfyui", "invert" }
local ANIMATIONS = { "idle" }
-- Models of the bridge's inpaint task, the first one is its default.
local INPAINT_MODELS = { "klein-4b", "sd15-inpaint" }
local MAX_ALERT_LINES = 15
local MAX_VARIANTS = 8
-- klein base was only measured up to a batch of 4 on 4 GB of VRAM.
local MAX_WALK_VARIANTS = 4
-- Each variant of an animation costs one klein redraw per pose.
local MAX_ANIMATION_VARIANTS = 4
-- Each variant of the rotations costs one klein drawing per view, side and back.
local MAX_ROTATE_VARIANTS = 4
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
-- The run in progress: only one generation at a time.
local running

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
  dlg:combobox{ id = "inpaint_model", label = "Inpaint model", option = prefs.inpaint_model,
                options = INPAINT_MODELS }
  dlg:entry{ id = "bridge_command", label = "Bridge command", text = prefs.bridge_command }
  dlg:number{ id = "margin", label = "Context margin (px)",
              text = tostring(prefs.margin), decimals = 0 }
  dlg:button{ id = "ok", text = "OK", focus = true }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if dlg.data.ok then
    prefs.backend = dlg.data.backend
    prefs.inpaint_model = dlg.data.inpaint_model
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

-- Returns false when the user cancels.
local function ask_character(prefs)
  local dlg = Dialog("AI Character")
  dlg:entry{ id = "prompt", label = "Description", text = prefs.character_prompt, focus = true }
  dlg:combobox{ id = "size", label = "Size (px)", option = prefs.character_size,
                options = CHARACTER_SIZES }
  dlg:check{ id = "use_palette", text = "Use the sprite palette", selected = prefs.use_palette }
  dlg:slider{ id = "variants", label = "Variants", min = 1, max = MAX_VARIANTS,
              value = prefs.variants }
  dlg:combobox{ id = "output", label = "Output", option = prefs.character_output,
                options = CHARACTER_OUTPUTS }
  dlg:button{ id = "ok", text = "Generate" }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if not dlg.data.ok then
    return false
  end
  if dlg.data.prompt:match("^%s*$") then
    app.alert{ title = "AI", text = "Describe the character to draw." }
    return false
  end
  prefs.character_prompt = dlg.data.prompt
  prefs.character_size = dlg.data.size
  prefs.use_palette = dlg.data.use_palette
  prefs.variants = dlg.data.variants
  prefs.character_output = dlg.data.output
  return true
end

-- Returns false when the user cancels.
local function ask_walk(prefs)
  local dlg = Dialog("AI Walk Sheet")
  dlg:check{ id = "from_sprite", text = "Animate the active sprite (or its selection)",
             selected = app.sprite ~= nil and prefs.walk_from_sprite, enabled = app.sprite ~= nil }
  dlg:entry{ id = "prompt", label = "Description", text = prefs.walk_prompt, focus = true }
  dlg:check{ id = "use_palette", text = "Use the sprite palette", selected = prefs.use_palette }
  dlg:slider{ id = "variants", label = "Variants", min = 1, max = MAX_WALK_VARIANTS,
              value = math.min(prefs.walk_variants, MAX_WALK_VARIANTS) }
  dlg:combobox{ id = "output", label = "Output", option = prefs.walk_output,
                options = WALK_OUTPUTS }
  dlg:label{ text = "About 1 min per variant." }
  dlg:button{ id = "ok", text = "Generate" }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if not dlg.data.ok then
    return false
  end
  if dlg.data.prompt:match("^%s*$") and not dlg.data.from_sprite then
    app.alert{ title = "AI", text = "Describe the character to animate." }
    return false
  end
  if app.sprite then
    prefs.walk_from_sprite = dlg.data.from_sprite
  end
  prefs.walk_prompt = dlg.data.prompt
  prefs.use_palette = dlg.data.use_palette
  prefs.walk_variants = dlg.data.variants
  prefs.walk_output = dlg.data.output
  return true
end

-- Returns false when the user cancels.
local function ask_animation(prefs)
  local dlg = Dialog("AI Animation")
  dlg:combobox{ id = "animation", label = "Animation", option = prefs.animation,
                options = ANIMATIONS }
  dlg:entry{ id = "prompt", label = "Details", text = prefs.animation_prompt, focus = true }
  dlg:slider{ id = "variants", label = "Variants", min = 1, max = MAX_ANIMATION_VARIANTS,
              value = math.min(prefs.animation_variants, MAX_ANIMATION_VARIANTS) }
  dlg:label{ text = "Animates the active sprite (or its selection)." }
  dlg:button{ id = "ok", text = "Generate" }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if not dlg.data.ok then
    return false
  end
  prefs.animation = dlg.data.animation
  prefs.animation_prompt = dlg.data.prompt
  prefs.animation_variants = dlg.data.variants
  return true
end

-- Returns false when the user cancels.
local function ask_rotate(prefs)
  local dlg = Dialog("AI Rotations")
  dlg:entry{ id = "prompt", label = "Details", text = prefs.rotate_prompt, focus = true }
  dlg:slider{ id = "variants", label = "Variants", min = 1, max = MAX_ROTATE_VARIANTS,
              value = math.min(prefs.rotate_variants, MAX_ROTATE_VARIANTS) }
  dlg:label{ text = "Turns the active sprite (or its selection), drawn facing down." }
  dlg:button{ id = "ok", text = "Generate" }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if not dlg.data.ok then
    return false
  end
  prefs.rotate_prompt = dlg.data.prompt
  prefs.rotate_variants = dlg.data.variants
  return true
end

local function common_options(prefs)
  return {
    bridge_command = prefs.bridge_command,
    backend = prefs.backend,
    use_palette = prefs.use_palette,
    variants = prefs.backend == "comfyui" and prefs.variants or 1,
  }
end

local function inpaint_options(prefs)
  local options = common_options(prefs)
  options.margin = prefs.margin
  options.model = prefs.inpaint_model
  options.prompt = prefs.prompt
  options.denoise = prefs.denoise / 100
  return options
end

-- Without a sprite open the character always goes to a new sprite.
local function character_output(prefs)
  return app.sprite and prefs.character_output or "New sprite"
end

local function character_options(prefs)
  local options = common_options(prefs)
  local size = tonumber(prefs.character_size) or 32
  options.prompt = prefs.character_prompt ~= "" and prefs.character_prompt or "a knight"
  options.size = Size(size, size)
  options.new_sprite = character_output(prefs) == "New sprite"
  return options
end

-- Without a sprite open the walk sheet always goes to a new sprite.
local function walk_output(prefs)
  return app.sprite and prefs.walk_output or "New sprite"
end

local function walk_options(prefs)
  local options = common_options(prefs)
  if prefs.backend == "comfyui" then
    options.variants = math.min(prefs.walk_variants, MAX_WALK_VARIANTS)
  end
  options.from_sprite = app.sprite ~= nil and prefs.walk_from_sprite
  options.prompt = prefs.walk_prompt
  if options.prompt == "" and not options.from_sprite then
    options.prompt = "a knight"
  end
  options.new_sprite = walk_output(prefs) == "New sprite"
  return options
end

local function animation_options(prefs)
  local options = common_options(prefs)
  if prefs.backend == "comfyui" then
    options.variants = math.min(prefs.animation_variants, MAX_ANIMATION_VARIANTS)
  end
  options.animation = prefs.animation
  options.prompt = prefs.animation_prompt
  return options
end

local function rotate_options(prefs)
  local options = common_options(prefs)
  if prefs.backend == "comfyui" then
    options.variants = math.min(prefs.rotate_variants, MAX_ROTATE_VARIANTS)
  end
  options.prompt = prefs.rotate_prompt
  return options
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
      text = text .. (job.layer_name == "AI inpaint" and " (nearly unchanged)" or " (empty)")
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

-- target is the layer to write into, or nil for a new layer; without a
-- sprite the result goes to a new sprite.
local function apply_job(sprite, frame, job, target)
  local choice, index = "one", 1
  if #job.patches > 1 then
    choice, index = pick_variant(job)
  end
  if job.walk then
    if choice then
      inpaint.apply_walk(sprite, job, choice == "one" and index or nil)
    end
  elseif job.animation then
    if choice then
      inpaint.apply_animation(sprite, job, choice == "one" and index or nil)
    end
  elseif not sprite then
    if choice then
      inpaint.apply_new_sprite(job, choice == "one" and index or nil)
    end
  elseif choice == "one" then
    apply_one(sprite, frame, job, index, target)
  elseif choice == "all" then
    inpaint.apply_all_layers(sprite, frame, job)
  end
end

-- One entry per bridge task: ask fills the preferences from the task's dialog
-- and returns false when the user cancels, start launches the bridge in the
-- background and returns a run for inpaint.poll, or nil and a message.
local function task_entries()
  return {
    {
      name = "inpaint",
      command_id = "AiInpaintSelection",
      title = "Inpaint Selection",
      enabled = function()
        return running == nil and app.sprite ~= nil and not app.sprite.selection.isEmpty
      end,
      ask = ask_inpaint,
      options = inpaint_options,
      output = function(prefs) return prefs.output end,
      start = inpaint.start,
    },
    {
      name = "character",
      command_id = "AiGenerateCharacter",
      title = "Generate Character",
      enabled = function()
        return running == nil
      end,
      ask = ask_character,
      options = character_options,
      output = character_output,
      start = inpaint.start,
    },
    {
      name = "walk-sheet",
      command_id = "AiGenerateWalkSheet",
      title = "Generate Walk Sheet",
      enabled = function()
        return running == nil
      end,
      ask = ask_walk,
      options = walk_options,
      output = walk_output,
      start = inpaint.start,
    },
    {
      name = "animation",
      command_id = "AiGenerateAnimation",
      title = "Generate Animation",
      enabled = function()
        return running == nil and app.sprite ~= nil
      end,
      ask = ask_animation,
      options = animation_options,
      output = function() return "New frames" end,
      start = inpaint.start,
    },
    {
      name = "rotate",
      command_id = "AiGenerateRotations",
      title = "Generate Rotations",
      enabled = function()
        return running == nil and app.sprite ~= nil
      end,
      ask = ask_rotate,
      options = rotate_options,
      output = function() return "New frames" end,
      start = inpaint.start,
    },
  }
end

-- Returns nil when the result can still be applied, otherwise a message.
local function apply_problem(run_state)
  if not run_state.sprite then
    return nil
  end
  if inpaint.is_closed(run_state.sprite) then
    return "The sprite was closed during the generation."
  end
  if run_state.frame_number > #run_state.sprite.frames then
    return "The frame was deleted during the generation."
  end
  if run_state.target then
    local ok, problem = pcall(function()
      return inpaint.active_layer_problem(run_state.target)
        or (run_state.target.sprite ~= run_state.sprite and "The layer was deleted." or nil)
    end)
    if not ok then
      return "The active layer was deleted during the generation."
    end
    return problem
  end
  return nil
end

local function on_result(run_state, job, message)
  if not job then
    if message then
      app.alert{ title = "AI", text = alert_lines(message) }
    end
    return
  end
  local problem = apply_problem(run_state)
  if problem then
    app.alert{ title = "AI", text = problem }
    return
  end
  local sprite = run_state.sprite
  if not sprite then
    apply_job(nil, nil, job)
    return
  end
  app.sprite = sprite
  apply_job(sprite, sprite.frames[run_state.frame_number], job, run_state.target)
end

-- Shows a progress window while the bridge runs in the background; Cancel
-- or closing the window stops the bridge.
local function wait_for(run, run_state)
  local started = os.time()
  local done = false
  local timer
  local dlg = Dialog{
    title = "AI",
    onclose = function()
      if not done then
        done = true
        timer:stop()
        inpaint.cancel(run)
        running = nil
        local stopper
        stopper = Timer{ interval = 0.5, ontick = function()
          if inpaint.poll(run) ~= nil then
            stopper:stop()
          end
        end }
        stopper:start()
      end
    end,
  }
  dlg:label{ id = "status", text = run_state.label }
  dlg:button{ id = "cancel", text = "Cancel", onclick = function() dlg:close() end }
  local function stop()
    done = true
    timer:stop()
    running = nil
    dlg:close()
  end
  local function tick()
    local finished, job_or_message = inpaint.poll(run)
    if finished == nil then
      dlg:modify{ id = "status", text = run_state.label .. " " .. os.time() - started .. " s" }
      return
    end
    stop()
    if finished then
      on_result(run_state, job_or_message)
    else
      on_result(run_state, nil, job_or_message)
    end
  end
  timer = Timer{
    interval = 0.25,
    ontick = function()
      local ok, problem = pcall(tick)
      if not ok then
        if not done then
          stop()
        end
        app.alert{ title = "AI", text = alert_lines("Unexpected error:\n" .. tostring(problem)) }
      end
    end,
  }
  dlg:show{ wait = false }
  timer:start()
end

local function run_task(plugin, task)
  if running then
    app.alert{ title = "AI", text = "A generation is already running." }
    return
  end
  local prefs = preferences(plugin)
  if prefs.backend == "comfyui" and not task.ask(prefs) then
    return
  end
  local sprite, frame = app.sprite, app.frame
  local output = task.output(prefs)
  local target
  if output == "Active layer" then
    local problem = inpaint.active_layer_problem(app.layer)
    if problem then
      app.alert{ title = "AI", text = problem }
      return
    end
    target = app.layer
  end
  local options = task.options(prefs)
  options.task = task.name
  local run, message = task.start(sprite, frame, options)
  if not run then
    app.alert{ title = "AI", text = alert_lines(message) }
    return
  end
  running = run
  wait_for(run, {
    sprite = run.sprite,
    frame_number = frame and frame.frameNumber or 1,
    target = target,
    label = task.title .. ": generating " .. options.variants .. " variant(s)...",
  })
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

function exit(plugin)
  if running then
    inpaint.cancel(running)
  end
end
