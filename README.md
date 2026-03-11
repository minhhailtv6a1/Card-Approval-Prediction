# Card Approval Prediction

## Setup Instructions

1. Clone the repository
2. Copy `.env.example` to `.env`
3. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your-actual-api-key
   ```
4. Install required packages:
   ```
   pip install python-dotenv langgraph langchain langchain-core langchain-openai
   ```

## Security Note
Never commit your `.env` file or expose your API keys in the code.
