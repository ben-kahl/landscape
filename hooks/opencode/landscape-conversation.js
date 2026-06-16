// OUTDATED / UNSUPPORTED: this hook targeted the removed push-based HTTP capture
// endpoints (POST /hooks/conversation-turn, /hooks/session-end) and the deleted
// landscape_capture_hook.py. Automatic capture is currently Claude Code only, via
// end-of-session transcript-pull (see hooks/claude-code/settings.example.json and
// `landscape ingest-transcript`). Do not use until an opencode transcript reader lands.
export const LandscapeConversation = async ({ $, directory }) => {
  // Resolve the capture script from the Landscape checkout, not the project
  // OpenCode happens to be running in, so this plugin works globally and in
  // other projects. Falls back to `directory` when run inside the repo itself.
  const home = process.env.LANDSCAPE_HOME ?? directory
  const script = `${home}/scripts/landscape_capture_hook.py`

  return {
    event: async ({ event }) => {
      if (event.type !== "message.updated" && event.type !== "message.part.updated") {
        return
      }

      await $`python3 ${script} opencode`.stdin(JSON.stringify({ event }))
    },
  }
}
