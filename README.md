# 🤖 AI Project Idea Recommender

A hybrid recommendation system that combines **TF-IDF similarity** with **AI-powered generation** using the **Cerebras Inference API** to help students discover relevant project ideas.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-idea-recommender.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/ketann13/student-project-idea-recommender?style=social)](https://github.com/ketann13/student-project-idea-recommender)

## ✨ Features

- 🔍 **Smart Search**: Find projects similar to your interests using TF-IDF and cosine similarity
- 🤖 **AI Generation**: Get fresh, creative project ideas powered by Cerebras Llama 3.1 8B model
- 🎯 **Advanced Filters**: Filter by domain, difficulty level, and number of recommendations
- ⭐ **Favorites System**: Save your favorite projects for later reference
- 📥 **Export Results**: Download recommendations as CSV
- 🎨 **Clean UI**: Built with Streamlit for a smooth, intuitive user experience
- ⚡ **Fast Performance**: Optimized TF-IDF vectorization with bigram support

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- A Cerebras API key (get one at [cloud.cerebras.ai](https://cloud.cerebras.ai))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ketann13/student-project-idea-recommender.git
cd student-project-idea-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables - Create a `.env` file:
```env
CERABUS_API_KEY=your_cerebras_api_key_here
```

4. Run the app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Live Demo

Visit the live app: **[Student Project Idea Recommender](https://project-idea-recommender.streamlit.app/)**

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── model.py                        # ML recommendation engine (TF-IDF + cosine similarity)
├── llm_client.py                   # Cerebras API wrapper
├── utils.py                        # Helper functions
├── data/
│   └── project_ideas_dataset_1000.csv  # Project dataset (1000+ ideas)
├── project-recommender/            # Modular project structure
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## 🎯 How It Works

1. **Text Processing**: User input and project descriptions are cleaned and normalized
2. **TF-IDF Vectorization**: Converts text to numerical vectors with unigram and bigram features
3. **Cosine Similarity**: Calculates similarity scores between user input and projects
4. **Filtering**: Applies domain, difficulty, and popularity filters
5. **AI Enhancement**: Optionally generates new project ideas using Cerebras Llama 3.1 8B
6. **Results Display**: Shows ranked recommendations with detailed information

## 🛠️ Technologies Used

- **Streamlit**: Interactive web UI framework
- **scikit-learn**: TF-IDF vectorization and cosine similarity computation
- **pandas**: Data manipulation and CSV handling
- **Cerebras Inference API**: AI-powered project generation with Llama 3.1 8B
- **python-dotenv**: Environment variable management
- **requests**: HTTP client for API calls
- **FPDF**: PDF export functionality

## 🔑 Getting Your Cerebras API Key

1. Visit [Cerebras Cloud](https://cloud.cerebras.ai)
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new API key
5. Add it to your `.env` file or Streamlit secrets

### For Streamlit Cloud Deployment:

1. Go to your app settings
2. Click on "Secrets"
3. Add:
```toml
CERABUS_API_KEY = "your_cerebras_api_key_here"
```

## 📊 Dataset

The dataset includes **1000+ project ideas** with:
- **Title**: Project name
- **Description**: Detailed project overview
- **Domain**: AI, Web Development, Data Science, Blockchain, IoT, etc.
- **Difficulty**: Beginner, Intermediate, Advanced
- **Skills Required**: Technologies and frameworks needed
- **Goals**: Learning objectives
- **Popularity Score**: Community interest metric
- **Year**: Project timeline

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - feel free to use this project for learning and development.

## 👨‍💻 Author

**Ketan**  
GitHub: [@ketann13](https://github.com/ketann13)

## 🌟 Acknowledgments

- Built with ❤️ using Streamlit
- Powered by Cerebras Inference API
- Inspired by the need to help students find meaningful projects

## 📈 Future Enhancements

- [ ] User authentication and personalized recommendations
- [ ] Project comparison feature
- [ ] Save and share recommendation links
- [ ] Integration with GitHub for project templates
- [ ] Multi-language support
- [ ] Dark mode theme

---

**⭐ If you find this project helpful, please star it on GitHub!**
