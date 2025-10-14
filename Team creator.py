import streamlit as st
import random

st.title("🏆 Team Generator (24 players into 6 teams of 4)")

st.write("Enter up to 24 names (one per line):")
names_input = st.text_area("Names", placeholder="Enter 24 names, one per line...")

if st.button("Generate Teams"):
    names = [name.strip() for name in names_input.split("\n") if name.strip()]
    
    if len(names) != 24:
        st.error(f"You entered {len(names)} names. Please enter exactly 24 names.")
    else:
        random.shuffle(names)
        num_teams = 6
        team_size = 4
        teams = [names[i:i + team_size] for i in range(0, len(names), team_size)]
        
        st.success("✅ Teams generated successfully!")
        
        for i, team in enumerate(teams, start=1):
            st.subheader(f"Team {i}")
            for player in team:
                st.write(f"- {player}")
