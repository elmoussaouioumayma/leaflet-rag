"""Small Gradio interface on top of the same pipeline used by the API."""

from __future__ import annotations

from .pipeline import LeafletAssistant


def build_ui(assistant: LeafletAssistant):
    import gradio as gr

    def respond(question: str) -> tuple[str, str]:
        if not question or len(question.strip()) < 3:
            return "Please type a question.", ""
        result = assistant.ask(question)
        sources = "\n\n".join(
            f"**[{c.number}] {c.doc}**, section {c.section} ({c.section_title}), score {c.score}\n> {c.text[:400]}"
            for c in result.citations
        )
        return f"{result.answer}\n\n_{result.disclaimer}_", sources or "_No sources used._"

    with gr.Blocks(title="Medicine Leaflet Assistant") as demo:
        gr.Markdown(
            "# Medicine Leaflet Assistant\nAsk what a medicine's package leaflet says. "
            "Answers cite the leaflet section. Not medical advice."
        )
        question = gr.Textbox(label="Question", placeholder="What are the common side effects of ...?")
        ask = gr.Button("Ask", variant="primary")
        answer = gr.Markdown()
        with gr.Accordion("Sources", open=True):
            sources = gr.Markdown()
        ask.click(respond, question, [answer, sources])
        question.submit(respond, question, [answer, sources])
    return demo


def main() -> None:
    build_ui(LeafletAssistant.from_index()).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
