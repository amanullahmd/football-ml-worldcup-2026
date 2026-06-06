"""FIFA World Cup 2026 — official draw groups and tournament metadata.

Source: dev-docs/docs3.md (drawn groups as per FIFA).
Hosts: USA, Canada, Mexico. 48 teams, 12 groups of 4, top 2 + 8 best 3rd advance.
"""

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOST_COUNTRIES = ["United States", "Canada", "Mexico"]

# Name aliases — left side is what appears in martj42/international_results,
# right side is the FIFA name used in GROUPS above. Extend if explore shows gaps.
NAME_ALIASES = {
    "Republic of Ireland": "Ireland",
    "United States": "United States",
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Cape Verde": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
}


def all_teams():
    return [t for teams in GROUPS.values() for t in teams]


def group_of(team):
    for g, teams in GROUPS.items():
        if team in teams:
            return g
    return None


def normalize(name):
    return NAME_ALIASES.get(name, name)
