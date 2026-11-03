local DEFAULT_PREFERENCES = {
  bridge_command = "ai-inpaint-bridge",
  margin = 8,
  backend = "comfyui",
  prompt = "",
  denoise = 80,
}
local BACKENDS = { "comfyui", "invert" }
local MAX_ALERT_LINES = 15

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
  local dlg = Dialog("AI Inpaint Settings")
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
local function ask_generation(prefs)
  local dlg = Dialog("AI Inpaint")
  dlg:entry{ id = "prompt", label = "Prompt", text = prefs.prompt, focus = true }
  dlg:slider{ id = "denoise", label = "Change strength (%)", min = 0, max = 100,
              value = prefs.denoise }
  dlg:button{ id = "ok", text = "Generate" }
  dlg:button{ id = "cancel", text = "Cancel" }
  dlg:show()
  if not dlg.data.ok then
    return false
  end
  prefs.prompt = dlg.data.prompt
  prefs.denoise = dlg.data.denoise
  return true
end

local function run_options(prefs)
  return {
    bridge_command = prefs.bridge_command,
    margin = prefs.margin,
    backend = prefs.backend,
    prompt = prefs.prompt,
    denoise = prefs.denoise / 100,
  }
end

function init(plugin)
  inpaint = dofile(app.fs.joinPath(plugin.path, "inpaint.lua"))

  plugin:newCommand{
    id = "AiInpaintSelection",
    title = "AI Inpaint Selection",
    group = "edit_fx",
    onenabled = function()
      return app.sprite ~= nil and not app.sprite.selection.isEmpty
    end,
    onclick = function()
      local prefs = preferences(plugin)
      if prefs.backend == "comfyui" and not ask_generation(prefs) then
        return
      end
      local ok, message = inpaint.run(app.sprite, app.frame, run_options(prefs))
      if not ok then
        app.alert{ title = "AI Inpaint", text = alert_lines(message) }
      end
    end,
  }

  plugin:newCommand{
    id = "AiInpaintSettings",
    title = "AI Inpaint Settings...",
    group = "edit_fx",
    onclick = function()
      show_settings(plugin)
    end,
  }
end
