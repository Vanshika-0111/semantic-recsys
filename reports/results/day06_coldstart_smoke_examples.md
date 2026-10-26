# Day 6 — cold-start examples (bge-small)

Pool = most-liked movies in the 30 days before onboarding; ranked by cos(profile, movie) + λ·prior. Settings per onboarding type: `{"seeds_1": {"pool": 50, "lambda": 0.5}, "seeds_3": {"pool": 50, "lambda": 1.0}, "seeds_5": {"pool": 50, "lambda": 0.5}, "text": {"pool": 50, "lambda": 0.1}, "text+seeds_1": {"pool": 50, "lambda": 0.2}, "text+seeds_3": {"pool": 50, "lambda": 0.5}, "text+seeds_5": {"pool": 50, "lambda": 0.2}}`

## Free-text onboarding — nothing but a sentence

'Now' = the end of the data (2023-10-26), so the pool is recent popular movies.

**“a mind-bending science fiction film about dreams and memory”** → Inception (2010); WALL·E (2008); Ex Machina (2015); Memento (2000); Back to the Future (1985); Arrival (2016); Interstellar (2014); The Silence of the Lambs (1991); The Martian (2015); The Matrix (1999)

**“heartwarming animated movie about friendship for the whole family”** → WALL·E (2008); Spirited Away (Sen to Chihiro no kamikakushi) (2001); Shrek (2001); The Incredibles (2004); Forrest Gump (1994); Back to the Future (1985); Good Will Hunting (1997); Ex Machina (2015); Spider-Man: Into the Spider-Verse (2018); The Lord of the Rings: The Return of the King (2003)

**“slow, sad drama about grief and losing a parent”** → Fight Club (1999); The Godfather (1972); The Silence of the Lambs (1991); Goodfellas (1990); Memento (2000); Gladiator (2000); Knives Out (2019); The Prestige (2006); Arrival (2016); Forrest Gump (1994)

**“gritty crime thriller with a heist that goes wrong”** → The Usual Suspects (1995); Inception (2010); Knives Out (2019); Pulp Fiction (1994); The Departed (2006); The Wolf of Wall Street (2013); Goodfellas (1990); Seven (a.k.a. Se7en) (1995); Catch Me If You Can (2002); The Shawshank Redemption (1994)

**“feel-good romantic comedy set in New York”** → Goodfellas (1990); The Godfather: Part II (1974); Forrest Gump (1994); Spider-Man: Across the Spider-Verse (2023); The Big Short (2015); The Shawshank Redemption (1994); The Usual Suspects (1995); Deadpool (2016); The Wolf of Wall Street (2013); Good Will Hunting (1997)

**“scary folk horror in an isolated village”** → Seven (a.k.a. Se7en) (1995); Spirited Away (Sen to Chihiro no kamikakushi) (2001); The Silence of the Lambs (1991); Parasite (2019); The Lord of the Rings: The Fellowship of the Ring (2001); Inglourious Basterds (2009); Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981); Memento (2000); Knives Out (2019); Fight Club (1999)

**“epic war movie based on a true story”** → The Lord of the Rings: The Two Towers (2002); The Lord of the Rings: The Return of the King (2003); Iron Man (2008); Star Wars: Episode V - The Empire Strikes Back (1980); Schindler's List (1993); Star Wars: Episode IV - A New Hope (1977); Inglourious Basterds (2009); The Matrix (1999); Catch Me If You Can (2002); The Big Short (2015)

**“funny superhero movie that doesn't take itself seriously”** → Spider-Man: Into the Spider-Verse (2018); Deadpool (2016); The Incredibles (2004); The Dark Knight (2008); Iron Man (2008); Guardians of the Galaxy (2014); Spider-Man: Across the Spider-Verse (2023); The Dark Knight Rises (2012); Batman Begins (2005); Parasite (2019)

## Real cold-start users (never in training)

Prior at each user's own onboarding moment. ✓ = rated ≥ 4 among their last 10 ratings.

### User 1264 — onboarded 2015-04-27

Onboarding picks (first 5 likes): The Shawshank Redemption (1994); The Godfather (1972); Schindler's List (1993); The Godfather: Part II (1974); One Flew Over the Cuckoo's Nest (1975)

Simulated questionnaire: *Drama, crime and war movies*

| # | 5 seeds | because you liked | ✓ | questionnaire text only | ✓ |
|---:|---|---|:-:|---|:-:|
| 1 | The Matrix (1999) | The Godfather: Part II (1974) |  | Django Unchained (2012) |  |
| 2 | The Dark Knight (2008) | The Godfather (1972) |  | Whiplash (2014) |  |
| 3 | The Departed (2006) | The Godfather: Part II (1974) |  | Pulp Fiction (1994) |  |
| 4 | The Imitation Game (2014) | Schindler's List (1993) |  | The Prestige (2006) |  |
| 5 | Inception (2010) | Schindler's List (1993) |  | The Imitation Game (2014) |  |
| 6 | Pulp Fiction (1994) | The Shawshank Redemption (1994) |  | Gone Girl (2014) |  |
| 7 | Gone Girl (2014) | The Shawshank Redemption (1994) |  | Fight Club (1999) |  |
| 8 | Fight Club (1999) | One Flew Over the Cuckoo's Nest (1975) |  | The Silence of the Lambs (1991) |  |
| 9 | The Silence of the Lambs (1991) | The Shawshank Redemption (1994) |  | Inglourious Basterds (2009) |  |
| 10 | A Beautiful Mind (2001) | One Flew Over the Cuckoo's Nest (1975) |  | V for Vendetta (2006) |  |

### User 170146 — onboarded 2006-06-02

Onboarding picks (first 5 likes): Chicken Run (2000); A Few Good Men (1992); Hot Shots! Part Deux (1993); Sling Blade (1996); The Lord of the Rings: The Return of the King (2003)

Simulated questionnaire: *Drama, comedy and action movies*

| # | 5 seeds | because you liked | ✓ | questionnaire text only | ✓ |
|---:|---|---|:-:|---|:-:|
| 1 | The Lord of the Rings: The Two Towers (2002) | The Lord of the Rings: The Return of the King (2003) |  | Star Wars: Episode V - The Empire Strikes Back (1980) |  |
| 2 | The Lord of the Rings: The Fellowship of the Ring (2001) | The Lord of the Rings: The Return of the King (2003) |  | Crash (2004) |  |
| 3 | Saving Private Ryan (1998) | Hot Shots! Part Deux (1993) |  | Fight Club (1999) |  |
| 4 | Star Wars: Episode VI - Return of the Jedi (1983) | The Lord of the Rings: The Return of the King (2003) |  | Star Wars: Episode VI - Return of the Jedi (1983) |  |
| 5 | Braveheart (1995) | The Lord of the Rings: The Return of the King (2003) |  | Pulp Fiction (1994) |  |
| 6 | Indiana Jones and the Last Crusade (1989) | Hot Shots! Part Deux (1993) |  | Star Wars: Episode IV - A New Hope (1977) |  |
| 7 | Star Wars: Episode IV - A New Hope (1977) | The Lord of the Rings: The Return of the King (2003) |  | Kill Bill: Vol. 1 (2003) |  |
| 8 | Harry Potter and the Goblet of Fire (2005) | The Lord of the Rings: The Return of the King (2003) |  | X-Men (2000) |  |
| 9 | Star Wars: Episode V - The Empire Strikes Back (1980) | The Lord of the Rings: The Return of the King (2003) |  | Braveheart (1995) |  |
| 10 | Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) | The Lord of the Rings: The Return of the King (2003) |  | Terminator 2: Judgment Day (1991) |  |

