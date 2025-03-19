import gradio as gr
from backend import flow

# Define the title and description
title = "Intelligent Q&A System"  # [1](@ref)
description = "Enter a question, and the AI will answer."  # [1](@ref)


hint_pipeline1 = """
**The pipeline flow is as follows:**
1. Answer the query.
2. If the answer is irrelevant, rewrite the query and answer again.
3. Next, search the internet for an answer.
4. Rerank RAG and web contexts and retrieve top 3 answers for the query.
5. Use a destilled LLM to answer the query if irrelevant.
"""

image_path1 = "assets/pipeline.png"

# Define the function to process the question and generate an answer
def generate_answer_page1(question):
    answer = flow(question)
    return f"### **Answer**\n\n{answer}"  # Return the answer as Markdown with bold title

# Define the UI layout
with gr.Blocks() as demo:
    css = """
    /* ... (CSS remains unchanged) ... */
    """

    demo.css = css

    with gr.Column(scale=1, min_width=800):
        gr.Markdown(f"## {title}", elem_classes="center")
        gr.Markdown(description, elem_classes="center")

    with gr.TabItem("Pipeline", id="pipeline1"):
        with gr.Row():
            with gr.Column():
                gr.Image(value=image_path1, elem_classes="image-container")
                gr.Markdown(hint_pipeline1, elem_classes="hint")
            with gr.Column():
                input1 = gr.Textbox(
                    label="Enter your question",
                    placeholder="What are the advantages of Deepseek R1?",
                    lines=2,
                    elem_classes="textbox"
                )
                submit1 = gr.Button("Submit", elem_classes="button")
                output1 = gr.Markdown(label="Answer", elem_classes="output-box")

        submit1.click(generate_answer_page1, inputs=[input1], outputs=output1)

# Launch the Gradio app
demo.launch(share=True)