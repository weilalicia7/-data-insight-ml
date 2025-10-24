import streamlit as st
import requests
import datetime
import pandas as pd
st.title("🎓 Mentor Matching System")

field = st.selectbox("Field of Study", ['Human and social sciences (Psychology, History, Sociology, etc.)',
       'IT, IS, Data, Web, Tech',
       'Commerce, Management, Economics, Management', 'Medical, Health',
       'Communication, Marketing, Advertising',
       'Administrative, Reception', 'Other', 'Architecture', 'Agri-food',
       'Agriculture, Natural and Life Sciences', 'Events',
       'Human Resources', 'Accounting, Finance',
       'Banking, Insurance and Finance', 'Energy, environment',
       'Basic sciences (mathematics, physics, chemistry, etc.)',
       'Planning, Urbanism, Geography', 'Hospitality, Catering, Tourism',
       'Electronics, Telecoms and Networks', 'Mechanics, Energy',
       'Chemistry, Process Engineering', 'Literary/economic',
       'Industrial, Production, Logistics', 'Social professions',
       'Law, Justice, Notary', 'LANGUAGES', 'Paramedical training',
       'Art, Design', 'Sport',
       'International Relations, Humanitarian Action', 'Real estate',
       'Audio-visual', 'Fashion, Textiles', 'Building, Public Works',
       'Political Science', 'Aeronautics - Space',
       'Culture, heritage, art history', 'Journalism, Publishing',
       'Education', 'Defense, Police, Gendarmerie', 'Letters',
       'Scientist', 'Well-being, Beauty'])


degree = st.selectbox("Degree", ['Bac+1', 'Bac+5 and plus', 'Bac+4', 'Bac+3', 'Bac+2'])
program = st.selectbox("Program", ["PP","PNP","EL"])
# needs = st.text_area("Needs", "to apply, perform, motivate")
needs = st.selectbox("Needs", ['[pro]', '[study]', '[pro,study]'])
registration_date = st.date_input(
    "Registration date",
    value=datetime.date.today(),   # default
    min_value=datetime.date(2020, 1, 1),  
    max_value=datetime.date(2025, 12, 31) 
)
desired_exchange_frequency = st.selectbox("Desired exchange frequency", ['Once per month', 'NA',
       'More than once per week', 'Once every two weeks (or Bi-weekly)',
       'Once per week'])
hobby = st.selectbox("Hobby", ['football', 'basketball', 'movie theater', 'martial', 'art',
       'handball', 'running', 'metal', 'beach', 'dance', 'video games',
       'novel', 'writing', 'cities', 'creation', 'singing', 'manga',
       'boxing', 'ski', 'muscle', 'cook', 'surf', 'bike', 'biography',
       'photo', 'athlete', 'philosophy', 'afro', 'meditation', 'tennis',
       'hike', 'backpacker', 'concerts', 'volleyball', 'skateboard',
       'makeup', 'animals', 'variety', 'campaign', 'games', 'rock', 'rap',
       'pastry shop', 'poetry', 'rugby', 'swimming', 'theater', 'blues',
       'pop', 'comics', 'hip hop', 'gym', 'thriller', 'winter', 'jazz',
       'videos', 'gardening', 'escalation ', 'randb', 'electro',
       'science fiction', 'camping', 'spoke', 'yoga', 'classic', 'Latin',
       'reggae', 'house', 'indie', 'snow', 'drunk', 'fishing', 'punk',
       'funk', 'petanque', 'ping-pong'])


# json123 = {
#   "top_k": 10,
#   "results": [
#     {"mentor_id": 63437, "workfield": "Consulting-Audit", "current_role": "Project Manager", "final_score": 0.87},
#     {"mentor_id": 63561, "workfield": "Human Resources", "current_role": "Senior Consultant", "final_score": 0.82}
#   ]
# }


mentors = pd.read_csv("mentors.csv")


def recommend_json(field, degree, program, needs, desired_exchange_frequency, hobby, top_k = 5):
    pred_df = pd.DataFrame({
    "mentor_id": [69538, 65914, 67270, 70320, 64242],
    "final_score": [0.842587, 0.831716, 0.830444, 0.829041, 0.823248]
    })

    merged = pred_df.merge(mentors, on="mentor_id", how="left")

    cols = ["mentor_id", "workfield", "current_role", "final_score"]
    top = merged[cols].sort_values("final_score", ascending=False)
    results = [
        {
            "mentor_id": int(r["mentor_id"]) if pd.notna(r["mentor_id"]) else None,
            "workfield": r["workfield"],
            "current_role": r["current_role"],
            "final_score": float(r["final_score"])
        }
        for _, r in top.iterrows()
    ]
    return {"top_k": int(top_k), "results": results}



if st.button("Recommend"):
    
    json1 = recommend_json(field, degree, program, needs, desired_exchange_frequency, hobby, top_k = 5)
    
    st.title("🎓 Mentor Recommendations")
    results = json1.get("results", [])


    if not results:
        st.warning("No recommendations available.")
    else:
    
        df = pd.DataFrame(results)

    
        df = df.rename(columns={
        "mentor_id": "Mentor ID",
        "workfield": "Work Field",
        "current_role": "Current Role",
        "final_score": "Match Score"
        })

    
        df["Match Score"] = df["Match Score"].map(lambda x: f"{x:.4f}")

    
        def highlight_row(row):
            color = "#e8f5e9" if float(row["Match Score"]) > 0.85 else "#f0f0f0"
            return ['background-color: {}'.format(color)] * len(row)

        styled_df = df.style.apply(highlight_row, axis=1)

    
        st.dataframe(styled_df, use_container_width=True)

    
        st.markdown("### 🧭 Top Mentors")
        for i, row in df.iterrows():
            st.markdown(f"""
        <div style="padding:10px; margin:6px 0; border-radius:10px; background:#f9f9f9; box-shadow:0 2px 5px rgba(0,0,0,0.05)">
        <b>🏅 Mentor ID:</b> {row['Mentor ID']}<br>
        <b>💼 Field:</b> {row['Work Field']}<br>
        <b>👤 Role:</b> {row['Current Role']}<br>
        <b>✨ Match Score:</b> <span style="color:#2e7d32; font-weight:bold;">{row['Match Score']}</span>
        </div>
        """, unsafe_allow_html=True)




    
    
