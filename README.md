# 🔍 Système RAG — Assistant Documentaire

Interface conversationnelle pour interroger des documents PDF en langage naturel.

**Stack** : LangChain · FAISS · multilingual-E5-large · Cross-Encoder · Gemini 2.0 Flash · Streamlit

---

## 🚀 Démo en ligne

👉 **[Ouvrir l'application](https://votre-app.streamlit.app)**

---

## 🏗️ Architecture

```
streamlit_app.py          ← Interface utilisateur
├── app/
│   ├── config.py         ← Tous les paramètres
│   ├── services/
│   │   ├── rag_engine.py     ← Moteur RAG (retriever + LLM)
│   │   └── vectorstore.py    ← Gestion FAISS + embeddings E5
│   ├── utils/
│   │   └── helpers.py        ← Nettoyage texte, formatage contexte
│   └── artifacts/            ← Index FAISS pré-construit (versionné)
└── build_index.py        ← Script offline pour (re)construire l'index
```

---

## ⚙️ Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/rag-system.git
cd rag-system
```

### 2. Créer l'environnement Python

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# ou : .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Configurer la clé API

```bash
cp .env.example .env
# Éditez .env et remplacez your_google_api_key_here par votre clé
```

Obtenez votre clé sur : https://aistudio.google.com/app/apikey

### 4. Construire l'index (si vous ajoutez vos propres PDF)

```bash
# Placez vos PDF dans le dossier Data/
mkdir Data
cp vos_documents/*.pdf Data/

# Construire le vectorstore
python build_index.py --data Data/ --out app/artifacts/
```

> Si vous utilisez les artifacts déjà versionnés dans le dépôt, **sautez cette étape**.

### 5. Lancer l'application

```bash
streamlit run streamlit_app.py
```

---

## ☁️ Déploiement sur Streamlit Community Cloud (gratuit)

> **La clé API n'est jamais exposée** — elle est injectée via les Secrets de Streamlit Cloud.

### Étapes

1. **Pusher le projet sur GitHub**
   ```bash
   git add .
   git commit -m "Initial RAG deployment"
   git push origin main
   ```

2. **Créer une app sur [share.streamlit.io](https://share.streamlit.io)**
   - Connectez votre compte GitHub
   - Sélectionnez le dépôt et `streamlit_app.py` comme point d'entrée

3. **Ajouter la clé API dans les Secrets**
   - Dans Streamlit Cloud : **Settings → Secrets**
   - Coller exactement :
     ```toml
     GOOGLE_API_KEY = "AIzaSy..."
     ```

4. **Déployer** — l'app sera accessible via une URL publique en 2-3 minutes.

---

## 📦 Ajouter de nouveaux documents

```bash
# 1. Ajouter les PDF dans Data/
cp nouveaux_docs/*.pdf Data/

# 2. Reconstruire l'index
python build_index.py

# 3. Commiter les nouveaux artifacts
git add app/artifacts/
git commit -m "Update vectorstore with new documents"
git push
```

Streamlit Cloud redéploie automatiquement après chaque push.

---

## 🔧 Paramètres avancés

Tout est dans `app/config.py` :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `chunk_size` | 512 | Taille des chunks en caractères |
| `chunk_overlap` | 128 | Chevauchement entre chunks |
| `embed_model` | `intfloat/multilingual-e5-large` | Modèle d'embedding |
| `retrieval_k` | 10 | Candidats MMR initiaux |
| `rerank_top_n` | 4 | Docs après reranking |
| `llm_model` | `gemini-2.0-flash` | Modèle LLM |

---

## 📄 Licence

MIT
