"""
SIMULATEUR WEB — CONGESTION RÉSEAU & ÉQUILIBRE DE NASH
Version Streamlit pour hébergement en ligne.

À lancer localement avec : streamlit run app.py
"""

import random
import math
from io import StringIO
import csv

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# =============================================================
# CONFIGURATION STREAMLIT
# =============================================================

st.set_page_config(
    page_title="Congestion réseau — Équilibre de Nash",
    page_icon="🌐",
    layout="wide",
)

# =============================================================
# STYLE
# =============================================================

PATH_COLORS = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#8b5cf6", "#14b8a6"]
BG_APP = "#f8fafc"
FG_TEXT = "#111827"
FG_BLUE = "#2563eb"
BORDER = "#d1d5db"

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #475569;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        margin-bottom: 10px;
    }
    .small-note {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================
# FONCTIONS DU JEU
# =============================================================

def cout_chemin(indice_chemin, delais_base, affectations, alpha):
    """Calcule le coût d’un chemin : C_j = d_j + alpha * x_j."""
    d_j = delais_base[indice_chemin]
    x_j = affectations.count(indice_chemin)
    return d_j + alpha * x_j


def tous_les_couts(delais_base, affectations, alpha):
    """Retourne les coûts de tous les chemins."""
    return [cout_chemin(j, delais_base, affectations, alpha) for j in range(len(delais_base))]


def cout_total_reseau(delais_base, affectations, alpha):
    """Coût total du réseau : somme des coûts payés par tous les nœuds."""
    total = 0.0
    for j, d_j in enumerate(delais_base):
        x_j = affectations.count(j)
        total += x_j * (d_j + alpha * x_j)
    return total


def initialiser_affectations(nb_noeuds, nb_chemins, graine=42):
    """Affectation aléatoire initiale des nœuds aux chemins."""
    random.seed(graine)
    return [random.randint(0, nb_chemins - 1) for _ in range(nb_noeuds)]


def calculer_equilibre_nash(nb_noeuds, delais_base, alpha, max_iterations=200, graine=42):
    """
    Algorithme de meilleure réponse.
    Chaque nœud change de chemin s’il trouve un coût plus faible.
    L’arrêt correspond à l’équilibre de Nash.
    """
    nb_chemins = len(delais_base)
    affectations = initialiser_affectations(nb_noeuds, nb_chemins, graine)
    historique = []
    convergence = None

    for iteration in range(max_iterations):
        historique.append({
            "iteration": iteration,
            "affectations": list(affectations),
            "couts": tous_les_couts(delais_base, affectations, alpha),
            "cout_total": cout_total_reseau(delais_base, affectations, alpha),
        })

        changements = 0
        for i in range(nb_noeuds):
            chemin_actuel = affectations[i]
            cout_actuel = cout_chemin(chemin_actuel, delais_base, affectations, alpha)
            meilleur_chemin = chemin_actuel
            meilleur_cout = cout_actuel

            for candidat in range(nb_chemins):
                if candidat == chemin_actuel:
                    continue
                affectations[i] = candidat
                cout_candidat = cout_chemin(candidat, delais_base, affectations, alpha)
                affectations[i] = chemin_actuel

                if cout_candidat < meilleur_cout:
                    meilleur_cout = cout_candidat
                    meilleur_chemin = candidat

            if meilleur_chemin != chemin_actuel:
                affectations[i] = meilleur_chemin
                changements += 1

        if changements == 0:
            convergence = iteration + 1
            historique.append({
                "iteration": iteration + 1,
                "affectations": list(affectations),
                "couts": tous_les_couts(delais_base, affectations, alpha),
                "cout_total": cout_total_reseau(delais_base, affectations, alpha),
            })
            break

    return historique, convergence


def verifier_nash(affectations, delais_base, alpha):
    """Vérifie qu’aucun nœud ne peut améliorer son coût en changeant seul."""
    affectations = list(affectations)
    nb_chemins = len(delais_base)

    for i, chemin_actuel in enumerate(affectations):
        cout_actuel = cout_chemin(chemin_actuel, delais_base, affectations, alpha)
        for candidat in range(nb_chemins):
            if candidat == chemin_actuel:
                continue
            affectations[i] = candidat
            cout_candidat = cout_chemin(candidat, delais_base, affectations, alpha)
            affectations[i] = chemin_actuel
            if cout_candidat < cout_actuel:
                return False
    return True


def solution_centralisee(nb_noeuds, delais_base, alpha):
    """Solution globale simple : minimiser le coût total par coût marginal."""
    nb_chemins = len(delais_base)
    repartition = [0] * nb_chemins

    for _ in range(nb_noeuds):
        couts_marginaux = [
            delais_base[j] + alpha * (2 * repartition[j] + 1)
            for j in range(nb_chemins)
        ]
        meilleur = couts_marginaux.index(min(couts_marginaux))
        repartition[meilleur] += 1

    cout_total = sum(
        repartition[j] * (delais_base[j] + alpha * repartition[j])
        for j in range(nb_chemins)
    )
    return repartition, cout_total


def lire_delais(texte, nb_chemins):
    """Lit les délais saisis par l’utilisateur ou génère des valeurs aléatoires."""
    brut = texte.strip()
    if brut:
        try:
            valeurs = [float(x.strip()) for x in brut.split(",")]
            if len(valeurs) >= nb_chemins:
                return sorted(valeurs[:nb_chemins]), None
            return None, f"Il faut au moins {nb_chemins} valeurs. Exemple : 2,5,8"
        except ValueError:
            return None, "Format invalide. Utilisez des nombres séparés par des virgules : 2,5,8"

    random.seed(42)
    return sorted([random.randint(2, 10) for _ in range(nb_chemins)]), None

# =============================================================
# FIGURES
# =============================================================

def dessiner_reseau(historique, delais_base, alpha):
    """Dessine la configuration finale du réseau à l’équilibre de Nash."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(BG_APP)
    ax.set_facecolor(BG_APP)
    ax.axis("off")

    etat = historique[-1]
    affectations = etat["affectations"]
    couts = etat["couts"]
    nb_noeuds = len(affectations)
    nb_chemins = len(delais_base)
    repartition = [affectations.count(j) for j in range(nb_chemins)]
    max_rep = max(repartition) if max(repartition) > 0 else 1

    cols = min(4, max(2, math.ceil(math.sqrt(nb_noeuds))))
    rows = math.ceil(nb_noeuds / cols)

    x_nodes = 0.25
    x_paths = 3.05
    x_dst = 4.55
    col_space = 0.55
    row_space = 0.70
    height = max(nb_chemins * 1.35 + 1.0, rows * row_space + 1.5)

    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.4, height + 0.5)

    ax.text(x_nodes + 0.7, height + 0.18, f"NŒUDS / JOUEURS ({nb_noeuds})",
            ha="center", va="bottom", color=FG_BLUE, fontsize=8, fontweight="bold")
    ax.text(x_paths, height + 0.18, f"CHEMINS / STRATÉGIES ({nb_chemins})",
            ha="center", va="bottom", color=FG_BLUE, fontsize=8, fontweight="bold")
    ax.text(x_dst, height + 0.18, "DESTINATION",
            ha="center", va="bottom", color=FG_BLUE, fontsize=8, fontweight="bold")

    positions = {}
    for i in range(nb_noeuds):
        col = i % cols
        row = i // cols
        x = x_nodes + col * col_space
        y = height - 0.55 - row * row_space
        positions[i] = (x, y)
        chemin = affectations[i]
        couleur = PATH_COLORS[chemin % len(PATH_COLORS)]
        cercle = plt.Circle((x, y), 0.20, color=couleur, zorder=5, alpha=0.96)
        ax.add_patch(cercle)
        ax.text(x, y, str(i + 1), ha="center", va="center", color="white",
                fontsize=6, fontweight="bold", zorder=6)

    y_dst = height / 2
    cercle_dst = plt.Circle((x_dst, y_dst), 0.36, facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2, zorder=5)
    ax.add_patch(cercle_dst)
    ax.text(x_dst, y_dst, "DST", ha="center", va="center", color=FG_TEXT,
            fontsize=9, fontweight="bold", zorder=6)

    # Cercles de chemins : taille fixe pour éviter les chevauchements
    rayon_chemin = 0.42
    for j in range(nb_chemins):
        y = height * (j + 1) / (nb_chemins + 1)
        couleur = PATH_COLORS[j % len(PATH_COLORS)]
        nb = repartition[j]
        ratio = nb / max_rep
        r, g, b = mcolors.to_rgb(couleur)
        teinte = (
            min(1, r + 0.10 * ratio),
            max(0, g * (1 - 0.18 * ratio)),
            max(0, b * (1 - 0.18 * ratio)),
        )

        cercle = plt.Circle((x_paths, y), rayon_chemin, color=teinte, zorder=4, alpha=0.96)
        ax.add_patch(cercle)
        ax.text(x_paths, y, f"C{j+1}\n{nb} nœuds\nCoût={couts[j]:.1f}",
                ha="center", va="center", color="white", fontsize=6.4, fontweight="bold", zorder=6)

        for i in range(nb_noeuds):
            if affectations[i] == j:
                x_n, y_n = positions[i]
                ax.plot([x_n + 0.20, x_paths - rayon_chemin], [y_n, y], color=couleur, alpha=0.14, lw=0.75, zorder=1)

        if nb > 0:
            ax.annotate("", xy=(x_dst - 0.36, y_dst), xytext=(x_paths + rayon_chemin, y),
                        arrowprops=dict(arrowstyle="-|>", color=couleur, lw=1.4 + 2.0 * ratio,
                                        connectionstyle="arc3,rad=0.08"), zorder=3)

    legendes = [mpatches.Patch(color=PATH_COLORS[j % len(PATH_COLORS)],
                               label=f"Chemin {j+1} (d={delais_base[j]})")
                for j in range(nb_chemins)]
    ax.legend(handles=legendes, loc="lower right", fontsize=7, frameon=True)

    ax.set_title(
        f"Configuration finale : équilibre de Nash à l’itération {etat['iteration']} | alpha={alpha:.2f}",
        fontsize=9, color=FG_TEXT, pad=8,
    )
    return fig


def dessiner_evolution_couts(historique, delais_base):
    """Évolution du coût des chemins jusqu’à l’équilibre."""
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#dbe3ef", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    iterations = [h["iteration"] for h in historique]
    for j in range(len(delais_base)):
        courbe = [h["couts"][j] for h in historique]
        ax.plot(iterations, courbe, marker="o", markersize=3.5, linewidth=1.8,
                color=PATH_COLORS[j % len(PATH_COLORS)], label=f"Chemin {j+1} (d={delais_base[j]})")

    ax.set_title("Évolution du coût des chemins", fontsize=10, color=FG_TEXT)
    ax.set_xlabel("Itération", fontsize=8)
    ax.set_ylabel("Coût Cj", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best")
    return fig


def dessiner_repartition(historique, delais_base):
    """Répartition finale des nœuds sur les chemins."""
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color="#dbe3ef", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    etat = historique[-1]
    affectations = etat["affectations"]
    nb_chemins = len(delais_base)
    repartition = [affectations.count(j) for j in range(nb_chemins)]
    labels = [f"C{j+1}\nd={delais_base[j]}" for j in range(nb_chemins)]
    couleurs = [PATH_COLORS[j % len(PATH_COLORS)] for j in range(nb_chemins)]

    barres = ax.bar(labels, repartition, color=couleurs, edgecolor="#475569", linewidth=0.7)
    for barre, val in zip(barres, repartition):
        ax.text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 0.05,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold", color=FG_TEXT)

    ax.set_title("Répartition finale des nœuds sur les chemins", fontsize=10, color=FG_TEXT)
    ax.set_ylabel("Nombre de nœuds", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    return fig


def generer_csv(historique, delais_base):
    """Prépare un CSV téléchargeable depuis Streamlit."""
    m = len(delais_base)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["iteration"] + [f"noeuds_chemin_{j+1}" for j in range(m)] +
                    [f"cout_chemin_{j+1}" for j in range(m)] + ["cout_total", "nash_final"])
    for h in historique:
        rep = [h["affectations"].count(j) for j in range(m)]
        writer.writerow([h["iteration"]] + rep + [f"{c:.4f}" for c in h["couts"]] +
                        [f"{h['cout_total']:.4f}", "oui" if h == historique[-1] else "non"])
    return output.getvalue()


def generer_resume(nb_noeuds, nb_chemins, alpha, delais_base, historique, convergence, cout_nash, cout_centralise, poa, repartition_centralisee):
    """Résumé prêt à copier dans le rapport."""
    final = historique[-1]
    affect = final["affectations"]
    rep = [affect.count(j) for j in range(nb_chemins)]
    lignes = [
        "RÉSUMÉ DES RÉSULTATS",
        "=" * 50,
        f"Nombre de nœuds : {nb_noeuds}",
        f"Nombre de chemins : {nb_chemins}",
        f"Facteur de congestion alpha : {alpha:.2f}",
        f"Délais de base : {delais_base}",
        f"Convergence vers Nash : itération {convergence}",
        "",
        "Répartition finale des nœuds :",
    ]
    for j in range(nb_chemins):
        lignes.append(f"- Chemin {j+1} : {rep[j]} nœuds, coût = {final['couts'][j]:.2f}")
    lignes += [
        "",
        f"Répartition centralisée : {repartition_centralisee}",
        f"Coût total à l’équilibre de Nash : {cout_nash:.2f}",
        f"Coût total de la solution centralisée : {cout_centralise:.2f}",
        f"Prix de l’anarchie (PoA) : {poa:.4f}",
        f"Perte d’efficacité : {(poa - 1) * 100:.1f}%",
        "",
        "Interprétation :",
        "L’équilibre de Nash représente une solution distribuée où chaque nœud choisit son meilleur chemin individuellement.",
        "La solution centralisée représente une solution globale optimisée.",
        "Si PoA = 1, il n’y a pas de perte d’efficacité. Si PoA > 1, les décisions individuelles entraînent une perte par rapport à l’optimum global.",
    ]
    return "\n".join(lignes)

# =============================================================
# INTERFACE STREAMLIT
# =============================================================

st.markdown('<div class="main-title">🌐 Simulateur de congestion réseau — Équilibre de Nash</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Application web interactive basée sur la théorie des jeux pour modéliser la congestion entre les nœuds.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Paramètres")
    nb_noeuds = st.slider("Nombre de nœuds (joueurs)", 2, 40, 12)
    nb_chemins = st.slider("Nombre de chemins (stratégies)", 2, 6, 3)
    alpha = st.slider("Facteur de congestion alpha", 0.1, 5.0, 1.5, step=0.1)
    delais_saisie = st.text_input("Délais de base (optionnel)", placeholder="Exemple : 2,5,8")
    st.caption("Si vous laissez vide, les délais sont générés automatiquement.")
    lancer = st.button("▶ Lancer la simulation", type="primary", use_container_width=True)

# La simulation se lance automatiquement au chargement et à chaque changement de paramètre
if not lancer:
    pass

delais_base, erreur = lire_delais(delais_saisie, nb_chemins)
if erreur:
    st.error(erreur)
    st.stop()

historique, convergence = calculer_equilibre_nash(nb_noeuds, delais_base, alpha, graine=42)
final = historique[-1]
cout_nash = final["cout_total"]
repartition_centralisee, cout_centralise = solution_centralisee(nb_noeuds, delais_base, alpha)
poa = cout_nash / cout_centralise if cout_centralise > 0 else 1.0
verifie = verifier_nash(final["affectations"], delais_base, alpha)
perte = (poa - 1) * 100

# Résultats principaux
col1, col2, col3, col4 = st.columns(4)
col1.metric("Convergence", f"Itération {convergence}")
col2.metric("Coût Nash", f"{cout_nash:.2f}")
col3.metric("Coût centralisé", f"{cout_centralise:.2f}")
col4.metric("Prix de l’anarchie", f"{poa:.4f}", delta=f"{perte:.1f}% perte")

if verifie:
    st.success("✅ Équilibre de Nash vérifié : aucun nœud ne peut améliorer son coût en changeant seul de chemin.")
else:
    st.warning("⚠️ L’état final n’est pas vérifié comme équilibre de Nash.")

# Visualisations
left, right = st.columns([1.25, 1])
with left:
    st.subheader("Configuration finale du réseau")
    fig_reseau = dessiner_reseau(historique, delais_base, alpha)
    st.pyplot(fig_reseau, use_container_width=True)

with right:
    st.subheader("Évolution et répartition")
    fig_couts = dessiner_evolution_couts(historique, delais_base)
    st.pyplot(fig_couts, use_container_width=True)
    fig_rep = dessiner_repartition(historique, delais_base)
    st.pyplot(fig_rep, use_container_width=True)

# Résumé pour le rapport
st.subheader("📋 Résumé du résultat")
resume = generer_resume(
    nb_noeuds,
    nb_chemins,
    alpha,
    delais_base,
    historique,
    convergence,
    cout_nash,
    cout_centralise,
    poa,
    repartition_centralisee,
)
st.text_area("", resume, height=260)

csv_data = generer_csv(historique, delais_base)
st.download_button(
    "💾 Télécharger les résultats CSV",
    data=csv_data,
    file_name="resultats_congestion_nash.csv",
    mime="text/csv",
)

with st.expander("ℹ️ Explication simple du modèle"):
    st.markdown(
        r"""
        - **Nœuds = joueurs** : chaque nœud prend une décision individuelle.
        - **Chemins = stratégies** : chaque joueur choisit un chemin.
        - **Coût du chemin** :

        $$C_j = d_j + \alpha x_j$$

        où $d_j$ est le délai de base, $x_j$ le nombre de nœuds sur le chemin, et $\alpha$ le facteur de congestion.
        - **Équilibre de Nash** : état stable où aucun nœud ne peut réduire son coût en changeant seul de chemin.
        - **Solution centralisée** : répartition globale qui minimise le coût total.
        - **Prix de l’anarchie (PoA)** : mesure la perte causée par les décisions individuelles.
        """
    )
