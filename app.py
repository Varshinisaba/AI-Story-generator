import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title("🧙‍♂️ AI Story Generator")
st.markdown("Enter a story prompt and let GPT-2 write an imaginative story!")

prompt = st.text_area("Your Story Prompt", "Once upon a time, in a distant galaxy,")

genre = st.selectbox("Choose a genre", ["Fantasy", "Sci-Fi", "Mystery", "Adventure", "Horror"])

length = st.slider("Story Length", min_value=50, max_value=500, value=150, step=10)

if st.button("Generate Story"):
    if not prompt.strip():
        st.error("Please enter a valid story prompt!")
    else:
        with st.spinner("Generating your story..."):
            prompt_with_genre = (
                f"You are a {genre} story writer. Continue the story with vivid descriptions:\n{prompt}"
            )
            input_ids = tokenizer.encode(prompt_with_genre, return_tensors="pt").to(device)
            output = model.generate(
                input_ids,
                max_length=length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
            story = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("📝 Generated Story")
            st.text_area("Your AI Story", value=story, height=300)
