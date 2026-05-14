export const LandscapeConversation = async ({ $, directory }) => {
  const script = `${directory}/scripts/landscape_capture_hook.py`

  return {
    event: async ({ event }) => {
      if (event.type !== "message.updated" && event.type !== "message.part.updated") {
        return
      }

      await $`python ${script} opencode`.stdin(JSON.stringify({ event }))
    },
  }
}
