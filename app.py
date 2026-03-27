import gradio as gr
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

def predict(*inputs):
    input_dict = dict(zip(features, inputs))
    input_df = pd.DataFrame([input_dict])

    # Convert numeric
    for col in input_df.columns:
        try:
            input_df[col] = pd.to_numeric(input_df[col])
        except:
            pass

    prediction = model.predict(input_df)[0]
    return f"Predicted G3: {prediction}"

inputs = [gr.Textbox(label=feature) for feature in features]

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="🎓 Student Performance Predictor"
)

interface.launch()
