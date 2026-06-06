Yes. The official draw has already been made, so we know the groups. ([FIFA][1])

## Groups

### Group A

* Mexico
* South Africa
* South Korea
* Czech Republic

### Group B

* Canada
* Bosnia and Herzegovina
* Qatar
* Switzerland

### Group C

* Brazil
* Morocco
* Haiti
* Scotland

### Group D

* United States
* Paraguay
* Australia
* Turkey

### Group E

* Germany
* Curacao
* Ivory Coast
* Ecuador

### Group F

* Netherlands
* Japan
* Sweden
* Tunisia

### Group G

* Belgium
* Egypt
* Iran
* New Zealand

### Group H

* Spain
* Cape Verde
* Saudi Arabia
* Uruguay

### Group I

* France
* Senegal
* Iraq
* Norway

### Group J

* Argentina
* Algeria
* Austria
* Jordan

### Group K

* Portugal
* DR Congo
* Uzbekistan
* Colombia

### Group L

* England
* Croatia
* Ghana
* Panama

([FIFA][1])

---

# Knockout Qualification

Advance to Round of 32:

```text
12 Group Winners
12 Group Runners-up
8 Best Third-Place Teams
= 32 Teams
```

([FIFA][1])

---

# How The Bracket Works

Some key Round-of-32 pairings are fixed by FIFA. Examples include: ([FIFA][2])

```text
A Runner-up vs B Runner-up

C Winner vs F Runner-up

F Winner vs C Runner-up

E Runner-up vs I Runner-up

A Winner vs Qualified Third-Place Team

E Winner vs Qualified Third-Place Team

I Winner vs Qualified Third-Place Team
```

The exact opponent for many group winners depends on **which 8 third-place teams qualify**, so the full bracket cannot be filled until the group stage ends. ([FIFA][2])

---

# For Your ML Project

Create these features:

```text
group
group_position

1st
2nd
3rd
4th

qualified_as
winner
runner_up
best_third

knockout_side
left
right

potential_path_to_final
```

Example:

```text
Brazil
→ Group C Winner
→ Round of 32
→ Round of 16
→ Quarterfinal
→ Semifinal
→ Final
```

---

The next thing you should build is a **World Cup 2026 bracket generator** inside your ML system.

Input:

```json
{
  "GroupA": ["Mexico", "South Korea"],
  "GroupB": ["Canada", "Switzerland"],
  ...
}
```

Output:

```json
{
  "round32": [...],
  "round16": [...],
  "quarterfinal": [...],
  "semifinal": [...],
  "final": [...]
}
```

This bracket engine is required before you can run Monte Carlo simulations and predict the tournament winner.

[1]: https://www.fifa.com/en/articles/groups-how-teams-qualify-tie-breakers?utm_source=chatgpt.com "World Cup 2026 groups, qualification rules & tie-breakers explained"
[2]: https://www.fifa.com/en/articles/knockout-stage-match-schedule-bracket?searchOverlay=1&utm_source=chatgpt.com "FIFA World Cup 2026 | Knockout stage match schedule bracket"
