import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==========================================
st.set_page_config(page_title="Classification - Régression Logistique", page_icon="🧠", layout="wide")

# ==========================================
# FONCTIONS MATHÉMATIQUES & MACHINE LEARNING
# ==========================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        # Éviter log(0) pour ne pas avoir d'erreur mathématique
        f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
        cost += -y[i]*np.log(f_wb) - (1-y[i])*np.log(1-f_wb)
    return cost / m

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

def compute_cost_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    cost_without_reg = compute_cost(X, y, w, b)
    reg_cost = sum(np.square(w))
    total_cost = cost_without_reg + (lambda_ / (2 * m)) * reg_cost
    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(num_iters):
        if lambda_ is not None:
            dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
            cost = cost_function(X, y, w, b, lambda_)
        else:
            dj_db, dj_dw = gradient_function(X, y, w, b)
            cost = cost_function(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost)

        # Mettre à jour l'interface tous les 10% de l'avancement
        if i % max(1, (num_iters // 10)) == 0 or i == num_iters - 1:
            progress_bar.progress((i + 1) / num_iters)
            status_text.text(f"Itération {i+1}/{num_iters} | Coût: {cost:.4f}")

    return w, b, J_history

def predict(X, w, b):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb >= 0.5 else 0
    return p

def map_feature(X1, X2):
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

# ==========================================
# FONCTIONS UTILITAIRES & GRAPHIQUES
# ==========================================
@st.cache_data
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    return X, y

def plot_decision_boundary(X, y, xlabel, ylabel, pos_label, neg_label, w=None, b=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data
    pos = y == 1
    neg = y == 0
    ax.scatter(X[pos, 0], X[pos, 1], marker='+', c='green', label=pos_label, s=80, linewidths=2)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', c='red', label=neg_label, s=80, edgecolors='k')

    # Plot decision boundary SEULEMENT si w et b sont fournis
    if w is not None and b is not None:
        if w.shape[0] <= 2:  # Cas linéaire
            plot_x = np.array([min(X[:, 0])-5, max(X[:, 0])+5])
            plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
            ax.plot(plot_x, plot_y, c="blue", linewidth=2, label="Frontière de décision")
            ax.set_xlim([min(X[:, 0])-5, max(X[:, 0])+5])
            ax.set_ylim([min(X[:, 1])-5, max(X[:, 1])+5])
        else:  # Cas polynomial
            u = np.linspace(-1.2, 1.5, 50)
            v = np.linspace(-1.2, 1.5, 50)
            z = np.zeros((len(u), len(v)))
            for i in range(len(u)):
                for j in range(len(v)):
                    # L'ERREUR ÉTAIT ICI : on force l'extraction du vecteur 1D avec [0]
                    mapped_point = map_feature(u[i], v[j])[0] 
                    z[i,j] = sigmoid(np.dot(mapped_point, w) + b)
            z = z.T
            ax.contour(u, v, z, levels=[0.5], colors="blue", linewidths=2)
            ax.plot([], [], color="blue", label="Frontière de décision")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return fig

# ==========================================
# APPLICATION PRINCIPALE STREAMLIT
# ==========================================
st.title("🧠 Classification par Régression Logistique")
st.markdown("""
Ce dashboard vous permet d'entraîner un modèle de machine learning, de visualiser sa **frontière de décision**, et de faire des **prédictions interactives** sur de nouvelles données.
""")

st.sidebar.header("⚙️ Paramètres de l'algorithme")
dataset_choice = st.sidebar.radio("Choisissez le cas d'usage :", 
                                  ("1. Admission Universitaire (Linéaire)", 
                                   "2. Assurance Qualité Puces (Régularisé)"))

st.sidebar.markdown("---")
alpha = st.sidebar.number_input("Taux d'apprentissage (Alpha)", value=0.001 if "1" in dataset_choice else 0.01, format="%.4f", step=0.001)
iterations = st.sidebar.slider("Nombre d'itérations", min_value=1000, max_value=50000, value=10000, step=1000)

# ==========================================
# CAS 1 : ADMISSION UNIVERSITAIRE
# ==========================================
if "1" in dataset_choice:
    st.header("🎓 Cas 1 : Prédiction d'Admission Universitaire")
    
    # Chargement des données
    try:
        X_train, y_train = load_data("/home/sitraka/Python_data/Projet_09_03_26/ex2data1.txt")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données. Veuillez vérifier le chemin du fichier. Détails: {e}")
        st.stop()
        
    tab1, tab2 = st.tabs(["📊 Entraînement & Visualisation", "🎯 Prédiction Interactive"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Données d'entraînement")
            fig_data = plot_decision_boundary(X_train, y_train, 'Score Examen 1', 'Score Examen 2', 'Admis', 'Non Admis')
            st.pyplot(fig_data)

        with col2:
            st.subheader("Entraîner le Modèle")
            st.markdown("Cliquez sur le bouton ci-dessous pour lancer la descente de gradient et trouver la meilleure séparation linéaire.")
            if st.button("🚀 Lancer l'entraînement", key="train_btn_1"):
                initial_w = np.zeros(X_train.shape[1])
                initial_b = 0.
                
                w_final, b_final, J_history = gradient_descent(X_train, y_train, initial_w, initial_b, 
                                                               compute_cost, compute_gradient, 
                                                               alpha, iterations)
                
                # Sauvegarde dans la session
                st.session_state['model_1_w'] = w_final
                st.session_state['model_1_b'] = b_final
                
                p = predict(X_train, w_final, b_final)
                accuracy = np.mean(p == y_train) * 100
                st.session_state['model_1_acc'] = accuracy
                st.session_state['J_hist_1'] = J_history
                
                st.success("Modèle entraîné avec succès !")

        # Affichage des résultats SI le modèle est entraîné
        if 'model_1_w' in st.session_state:
            st.divider()
            col3, col4, col5 = st.columns([1, 1, 1])
            col3.metric("Précision (Accuracy)", f"{st.session_state['model_1_acc']:.2f} %")
            
            with col4:
                st.subheader("Frontière de décision")
                fig_bound = plot_decision_boundary(X_train, y_train, 'Score Examen 1', 'Score Examen 2', 'Admis', 'Non Admis', 
                                                   w=st.session_state['model_1_w'], b=st.session_state['model_1_b'])
                st.pyplot(fig_bound)
                
            with col5:
                st.subheader("Convergence du Coût")
                fig_cost, ax_cost = plt.subplots(figsize=(8, 6))
                ax_cost.plot(st.session_state['J_hist_1'])
                ax_cost.set_xlabel("Itérations")
                ax_cost.set_ylabel("Coût J")
                st.pyplot(fig_cost)
                
            st.info("💡 Allez dans l'onglet 'Prédiction Interactive' (en haut) pour tester le modèle !")

    with tab2:
        st.subheader("Testez un nouvel étudiant")
        if 'model_1_w' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord entraîner le modèle dans l'onglet précédent.")
        else:
            c1, c2 = st.columns(2)
            exam1 = c1.number_input("Score Examen 1", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            exam2 = c2.number_input("Score Examen 2", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            
            if st.button("🎯 Prédire l'admission", key="pred_btn_1"):
                x_new = np.array([exam1, exam2])
                z = np.dot(x_new, st.session_state['model_1_w']) + st.session_state['model_1_b']
                prob = sigmoid(z)
                
                st.divider()
                if prob >= 0.5:
                    st.success(f"🎉 **ADMIS !** Cet étudiant a de grandes chances d'être accepté.")
                    st.metric("Probabilité d'admission", f"{prob * 100:.2f} %")
                else:
                    st.error(f"❌ **NON ADMIS.** Les scores de cet étudiant sont probablement insuffisants.")
                    st.metric("Probabilité d'admission", f"{prob * 100:.2f} %")

# ==========================================
# CAS 2 : ASSURANCE QUALITÉ PUCES
# ==========================================
else:
    st.header("💻 Cas 2 : Test Qualité Micro-puces (Régularisation)")
    lambda_ = st.sidebar.slider("Paramètre de Régularisation (Lambda)", min_value=0.0, max_value=10.0, value=0.01, step=0.01)
    
    try:
        X_train, y_train = load_data("/home/sitraka/Python_data/Projet_09_03_26/ex2data2.txt")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données. Détails: {e}")
        st.stop()
        
    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    
    tab1, tab2 = st.tabs(["📊 Entraînement & Visualisation", "🎯 Prédiction Interactive"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Données d'entraînement")
            fig_data = plot_decision_boundary(X_train, y_train, 'Test Puce 1', 'Test Puce 2', 'Acceptée', 'Rejetée')
            st.pyplot(fig_data)
            st.caption("Remarquez que les données ne peuvent pas être séparées par une simple ligne droite.")

        with col2:
            st.subheader("Entraîner le Modèle (Polynomial)")
            st.markdown(f"Nous générons des caractéristiques polynomiales jusqu'au **degré 6** (soit {X_mapped.shape[1]} features).")
            if st.button("🚀 Lancer l'entraînement", key="train_btn_2"):
                np.random.seed(1)
                initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
                initial_b = 1.
                
                w_final, b_final, J_history = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                                               compute_cost_reg, compute_gradient_reg, 
                                                               alpha, iterations, lambda_)
                
                st.session_state['model_2_w'] = w_final
                st.session_state['model_2_b'] = b_final
                
                p = predict(X_mapped, w_final, b_final)
                accuracy = np.mean(p == y_train) * 100
                st.session_state['model_2_acc'] = accuracy
                st.session_state['J_hist_2'] = J_history
                
                st.success("Modèle régularisé entraîné avec succès !")

        if 'model_2_w' in st.session_state:
            st.divider()
            col3, col4, col5 = st.columns([1, 1, 1])
            col3.metric("Précision (Accuracy)", f"{st.session_state['model_2_acc']:.2f} %")
            
            with col4:
                st.subheader("Frontière de décision (Non-linéaire)")
                fig_bound = plot_decision_boundary(X_train, y_train, 'Test Puce 1', 'Test Puce 2', 'Acceptée', 'Rejetée', 
                                                   w=st.session_state['model_2_w'], b=st.session_state['model_2_b'])
                st.pyplot(fig_bound)
                
            with col5:
                st.subheader("Convergence du Coût")
                fig_cost, ax_cost = plt.subplots(figsize=(8, 6))
                ax_cost.plot(st.session_state['J_hist_2'])
                ax_cost.set_xlabel("Itérations")
                ax_cost.set_ylabel("Coût J (Régularisé)")
                st.pyplot(fig_cost)
                
            st.info("💡 Allez dans l'onglet 'Prédiction Interactive' pour tester une puce avec des valeurs personnalisées !")

    with tab2:
        st.subheader("Testez une nouvelle micro-puce")
        if 'model_2_w' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord entraîner le modèle dans l'onglet précédent.")
        else:
            st.markdown("Insérez les résultats des tests (généralement entre -1.5 et 1.5) :")
            c1, c2 = st.columns(2)
            test1 = c1.slider("Score Test 1", min_value=-1.5, max_value=1.5, value=0.0, step=0.01)
            test2 = c2.slider("Score Test 2", min_value=-1.5, max_value=1.5, value=0.0, step=0.01)
            
            if st.button("🎯 Vérifier la puce", key="pred_btn_2"):
                # Feature mapping pour le point unique [0] pour sortir le vecteur 1D
                x_mapped_new = map_feature(test1, test2)[0] 
                z = np.dot(x_mapped_new, st.session_state['model_2_w']) + st.session_state['model_2_b']
                prob = sigmoid(z)
                
                st.divider()
                if prob >= 0.5:
                    st.success(f"✅ **PUCE VALIDÉE !** Cette micro-puce passe les contrôles de qualité.")
                    st.metric("Confiance du modèle", f"{prob * 100:.2f} %")
                else:
                    st.error(f"❌ **PUCE DÉFECTUEUSE.** Cette micro-puce doit être rejetée.")
                    st.metric("Probabilité d'être valide", f"{prob * 100:.2f} %")