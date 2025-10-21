# 🤖 AI Project Idea Recommender

A hybrid recommendation system that combines **TF-IDF similarity** with **AI-powered generation** using the Cerabus API to help students discover relevant project ideas.

## ✨ Features

- 🔍 **Smart Search**: Find projects similar to your interests using TF-IDF and cosine similarity
- 🤖 **AI Generation**: Get fresh, creative project ideas from Cerabus LLM API
- 🎯 **Advanced Filters**: Filter by domain, difficulty level, and more
- ⭐ **Favorites System**: Save your favorite projects for later
- 📥 **Export Results**: Download recommendations as CSV
- 🎨 **Clean UI**: Built with Streamlit for a smooth user experience

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ketann13/student-project-idea-recommender.git
cd "Project Recommendation System"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables - Create a `.env` file:
```env
CERABUS_API_KEY=your_cerabus_api_key_here
```

4. Run the app:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── model.py                        # ML recommendation engine
├── llm_client.py                   # Cerabus API wrapper
├── utils.py                        # Helper functions
├── data/
│   └── project_ideas_dataset_1000.csv  # Project dataset
├── project-recommender/            # Modular project structure
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Technologies Used

- **Streamlit**: Web UI framework
- **scikit-learn**: TF-IDF vectorization and similarity
- **pandas**: Data manipulation
- **Cerabus API**: AI-powered project generation
- **python-dotenv**: Environment variable management

## 🔑 API Keys

To enable AI-powered features, you need a Cerabus API key:
1. Sign up at [Cerabus](https://cerabus.com)
2. Get your API key
3. Add it to your `.env` file

## 👨‍💻 Author

**Ketan**  
GitHub: [@ketann13](https://github.com/ketann13)

---

**⭐ If you find this project helpful, please star it on GitHub!**
