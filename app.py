from processing import feature_extraction, preprocessing_feature, classified_feature
import gradio as gr

def predict_audiogram(filename, image_dir=""):
    # extract data from image
    extract_rt_df, extract_lt_df = feature_extraction(image_dir, filename)
    # preprocess data
    feature_rt_df, feature_lt_df = preprocessing_feature(extract_rt_df, extract_lt_df)
    predict = classified_feature(feature_rt_df, feature_lt_df)
    predict = predict.reset_index()
    predcit = predict.rename(columns = {'index':'side'})
    
    return predict


demo = gr.Interface(
    fn=predict_audiogram,
    inputs= gr.Image(type='filepath'),
    outputs="dataframe",
    description="Upload an image",
    title="Audiogram Prediction",
    theme="huggingface"
)

if __name__ == "__main__":
    demo.launch() 
