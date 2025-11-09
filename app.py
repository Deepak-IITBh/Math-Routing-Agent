import streamlit as st
import requests
import json

# Configure page
st.set_page_config(
    page_title="Math Routing Agent",
    page_icon="🧮",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Title and description
st.title("🧮 Math Routing Agent")
st.markdown("""
This application helps solve mathematical problems step by step.
Ask any math question, and the system will provide a detailed solution!
""")

# Input section
with st.form("math_form"):
    question = st.text_area("Enter your math question:", height=100)
    submitted = st.form_submit_button("Solve!")

    if submitted and question:
        try:
            # Call the FastAPI backend
            response = requests.post(
                "http://localhost:8000/query",
                json={"question": question, "user_id": "anonymous"}
            )
            
            if response.status_code == 200:
                result = response.json()
                # Add to history
                st.session_state.history.append({
                    "question": question,
                    "solution": result["solution"],
                    "steps": result["steps"],
                    "source": result["source"],
                    "confidence": result["confidence"],
                    "feedback": None
                })
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error connecting to the server: {str(e)}")

# History and feedback section
st.markdown("### Previous Questions and Solutions")

for idx, item in enumerate(reversed(st.session_state.history)):
    with st.expander(f"Question: {item['question'][:100]}..."):
        st.markdown("#### Solution:")
        st.markdown(item["solution"])
        
        st.markdown("#### Steps:")
        for i, step in enumerate(item["steps"], 1):
            st.markdown(f"{i}. {step}")
            
        st.markdown(f"*Source: {item['source']} (Confidence: {item['confidence']:.2f})*")
        
        # Feedback section
        if item["feedback"] is None:
            feedback = st.text_area("Provide feedback on this solution:", key=f"feedback_{idx}")
            if st.button("Submit Feedback", key=f"submit_{idx}"):
                try:
                    response = requests.post(
                        "http://localhost:8000/feedback",
                        json={
                            "question": item["question"],
                            "feedback": feedback
                        }
                    )
                    if response.status_code == 200:
                        item["feedback"] = feedback
                        st.success("Thank you for your feedback!")
                    else:
                        st.error(f"Error submitting feedback: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Error connecting to the server: {str(e)}")
        else:
            st.markdown("#### Your Feedback:")
            st.markdown(item["feedback"])

# Sidebar with info
st.sidebar.markdown("""
### About
This Math Routing Agent uses:
- Knowledge Base for common problems
- Web Search for new questions
- Human feedback for continuous improvement

### Features
- Step-by-step solutions
- Educational content focus
- Feedback mechanism
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Math Routing Agent Team")