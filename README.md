# HumanJOBS - Assistant RH Multi-Agents

## 🚀 Description

HumanJOBS est un système intelligent d'aide aux ressources humaines spécialement conçu pour le Maroc. Cette application utilise l'intelligence artificielle multi-agents pour fournir des solutions complètes en matière de gestion des ressources humaines, recrutement, et support RH.

## ✨ Fonctionnalités

- 🤖 **Assistant IA Multi-Agents** - Système intelligent basé sur plusieurs agents spécialisés
- 💼 **Gestion RH** - Outils complets pour la gestion des ressources humaines
- 🇲🇦 **Adapté au Maroc** - Solutions personnalisées pour le contexte marocain
- 🔍 **Recherche intelligente** - Recherche avancée avec DuckDuckGo
- 📊 **Interface moderne** - Interface utilisateur intuitive et responsive
- 🌐 **API RESTful** - Architecture moderne avec FastAPI

## 🛠️ Technologies Utilisées

- **Backend**: FastAPI, Flask
- **IA/ML**: LangChain, Google Generative AI, Ollama
- **Base de données**: ChromaDB, SQLite3
- **Frontend**: HTML, CSS, JavaScript
- **Autres**: BeautifulSoup4, MQTT, Pydantic

## 📋 Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)
- Git

## 🔧 Installation

### 1. Cloner le repository
```bash
git clone https://github.com/OthmanBouysfi/CHATBOOT-HUMANJOBS.git
cd CHATBOOT-HUMANJOBS
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv

# Sur Windows
venv\Scripts\activate

# Sur macOS/Linux
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Installer les dépendances supplémentaires
```bash
pip install fastapi
pip install google-generativeai
```

## 🚀 Lancement de l'application

### Démarrer le serveur de développement
```bash
uvicorn api:app --reload
```

L'application sera accessible à l'adresse: `http://localhost:8000`

## 📁 Structure du projet

```
CHATBOOT-HUMANJOBS/
├── api.py                 # Point d'entrée FastAPI
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation
├── static/               # Fichiers statiques (CSS, JS, images)
├── templates/            # Templates HTML
└── config/               # Fichiers de configuration
```

## ⚙️ Configuration

### Variables d'environnement
Créez un fichier `.env` à la racine du projet :

```env
# Configuration Google AI
GOOGLE_API_KEY=votre_clé_api_google

# Configuration Ollama
OLLAMA_HOST=http://localhost:11434

# Configuration de l'application
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

### Configuration Ollama (optionnel)
Si vous utilisez Ollama, assurez-vous qu'il est installé et en cours d'exécution :
```bash
ollama serve
```

## 🔧 Commandes utiles

### Développement
```bash
# Lancer en mode développement
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Lancer avec des logs détaillés
uvicorn api:app --reload --log-level debug
```

### Tests
```bash
# Installer les dépendances de test
pip install pytest pytest-asyncio

# Lancer les tests
pytest
```

## 📚 Documentation API

Une fois l'application lancée, vous pouvez accéder à la documentation interactive de l'API :

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commitez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Othman BOUYSFI** - *Tech Lead* - [@OthmanBouysfi](https://github.com/OthmanBouysfi)

## 🐛 Signaler un bug

Si vous trouvez un bug, veuillez ouvrir une [issue](https://github.com/OthmanBouysfi/CHATBOOT-HUMANJOBS/issues) avec :
- Une description claire du problème
- Les étapes pour reproduire le bug
- Votre environnement (OS, version Python, etc.)

## 📞 Support

Pour toute question ou support :
- Ouvrez une issue sur GitHub
- Contactez-moi via mon profil GitHub

---

![image](https://github.com/user-attachments/assets/c556866e-cf63-4bf4-8ec2-77ef7fa11424)
