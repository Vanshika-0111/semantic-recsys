# Day 6 — cold-start examples (bge-small)

Pool = most-liked movies overall; ranked by cos(profile, movie) + λ·prior. Accuracy-tuned settings: `{"seeds_1": {"pool": 20, "lambda": 2.0}, "seeds_3": {"pool": 20, "lambda": 5.0}, "seeds_5": {"pool": 20, "lambda": 5.0}, "text": {"pool": 20, "lambda": 1000.0}, "text+seeds_1": {"pool": 20, "lambda": 5.0}, "text+seeds_3": {"pool": 20, "lambda": 5.0}, "text+seeds_5": {"pool": 20, "lambda": 5.0}}`

## Free text — explore mode (semantic search, not accuracy-tuned)

The accuracy-tuned text setting is λ = 1000 (text added no measurable NDCG), which returns the same popular list for every sentence. Explore mode = pool of the 1,000 most-liked movies ranked by cos(text, movie) + 0.3·prior.

**“a mind-bending science fiction film about dreams and memory”** → Inception (2010); Total Recall (1990); Eternal Sunshine of the Spotless Mind (2004); The Butterfly Effect (2004); A Beautiful Mind (2001); Vanilla Sky (2001); Brazil (1985); Dark City (1998); Mulholland Drive (2001); Her (2013)

**“heartwarming animated movie about friendship for the whole family”** → Toy Story (1995); Bambi (1942); Big Hero 6 (2014); Lilo & Stitch (2002); Finding Nemo (2003); WALL·E (2008); Spirited Away (Sen to Chihiro no kamikakushi) (2001); Inside Out (2015); The Lego Movie (2014); Big Fish (2003)

**“slow, sad drama about grief and losing a parent”** → The Sixth Sense (1999); Magnolia (1999); The Royal Tenenbaums (2001); Silver Linings Playbook (2012); Hunt, The (Jagten) (2012); Dead Man Walking (1995); Hamlet (1996); Secrets & Lies (1996); Mrs. Doubtfire (1993); American Beauty (1999)

**“gritty crime thriller with a heist that goes wrong”** → The Usual Suspects (1995); Reservoir Dogs (1992); Inception (2010); Ocean's Eleven (2001); Fargo (1996); Drive (2011); L.A. Confidential (1997); Ocean's Thirteen (2007); Out of Sight (1998); Lock, Stock & Two Smoking Barrels (1998)

**“feel-good romantic comedy set in New York”** → Manhattan (1979); Breakfast at Tiffany's (1961); When Harry Met Sally... (1989); The Apartment (1960); As Good as It Gets (1997); The Terminal (2004); Notting Hill (1999); American Beauty (1999); Singin' in the Rain (1952); Trading Places (1983)

**“scary folk horror in an isolated village”** → Sleepy Hollow (1999); Poltergeist (1982); The Others (2001); The Evil Dead (1981); My Neighbor Totoro (Tonari no Totoro) (1988); The Shining (1980); The Nightmare Before Christmas (1993); The Exorcist (1973); An American Werewolf in London (1981); The Cabin in the Woods (2012)

**“epic war movie based on a true story”** → 300 (2007); Patton (1970); Saving Private Ryan (1998); Glory (1989); Black Hawk Down (2001); Enemy at the Gates (2001); Braveheart (1995); Fury (2014); The Bridge on the River Kwai (1957); Full Metal Jacket (1987)

**“funny superhero movie that doesn't take itself seriously”** → Napoleon Dynamite (2004); Kick-Ass (2010); Superman (1978); Untitled Spider-Man Reboot (2017); Spider-Man: Into the Spider-Verse (2018); Ant-Man (2015); The Avengers (2012); Avengers: Infinity War - Part I (2018); The Incredibles (2004); Austin Powers: The Spy Who Shagged Me (1999)

## Real cold-start users (never in training)

Prior at each user's own onboarding moment. ✓ = rated ≥ 4 among their last 10 ratings.

### User 966 — onboarded 2001-11-10

Onboarding picks (first 5 likes): Schindler's List (1993); Shanghai Noon (2000); Mystery, Alaska (1999); The Princess Bride (1987); Kiss Me, Guido (1997)

Simulated questionnaire: *Comedy, drama and action movies*

| # | 5 seeds | because you liked | ✓ | questionnaire text only | ✓ |
|---:|---|---|:-:|---|:-:|
| 1 | The Shawshank Redemption (1994) | Schindler's List (1993) |  | The Shawshank Redemption (1994) |  |
| 2 | Pulp Fiction (1994) | Kiss Me, Guido (1997) |  | Pulp Fiction (1994) |  |
| 3 | Forrest Gump (1994) | Schindler's List (1993) |  | The Matrix (1999) |  |
| 4 | The Matrix (1999) | Schindler's List (1993) |  | Forrest Gump (1994) |  |
| 5 | Fight Club (1999) | Kiss Me, Guido (1997) |  | The Silence of the Lambs (1991) |  |
| 6 | The Godfather (1972) | Kiss Me, Guido (1997) |  | Star Wars: Episode IV - A New Hope (1977) |  |
| 7 | The Usual Suspects (1995) | Mystery, Alaska (1999) | ✓ | Fight Club (1999) |  |
| 8 | The Lord of the Rings: The Fellowship of the Ring (2001) | The Princess Bride (1987) |  | Star Wars: Episode V - The Empire Strikes Back (1980) |  |
| 9 | American Beauty (1999) | Kiss Me, Guido (1997) |  | The Lord of the Rings: The Fellowship of the Ring (2001) |  |
| 10 | Braveheart (1995) | The Princess Bride (1987) |  | The Usual Suspects (1995) | ✓ |

### User 170907 — onboarded 2014-07-02

Onboarding picks (first 5 likes): Starship Troopers (1997); Desperado (1995); The People vs. Larry Flynt (1996); Little Shop of Horrors (1986); Roger & Me (1989)

Simulated questionnaire: *Action, comedy and science fiction movies*

| # | 5 seeds | because you liked | ✓ | questionnaire text only | ✓ |
|---:|---|---|:-:|---|:-:|
| 1 | Pulp Fiction (1994) | Desperado (1995) |  | The Shawshank Redemption (1994) |  |
| 2 | The Matrix (1999) | Starship Troopers (1997) |  | Pulp Fiction (1994) |  |
| 3 | The Shawshank Redemption (1994) | The People vs. Larry Flynt (1996) |  | The Matrix (1999) |  |
| 4 | The Silence of the Lambs (1991) | Little Shop of Horrors (1986) |  | Forrest Gump (1994) |  |
| 5 | Star Wars: Episode V - The Empire Strikes Back (1980) | Starship Troopers (1997) |  | The Silence of the Lambs (1991) |  |
| 6 | Star Wars: Episode IV - A New Hope (1977) | Starship Troopers (1997) |  | Star Wars: Episode IV - A New Hope (1977) |  |
| 7 | Fight Club (1999) | Desperado (1995) |  | Fight Club (1999) |  |
| 8 | The Usual Suspects (1995) | Desperado (1995) |  | Schindler's List (1993) |  |
| 9 | Terminator 2: Judgment Day (1991) | Starship Troopers (1997) |  | Star Wars: Episode V - The Empire Strikes Back (1980) |  |
| 10 | Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) | Roger & Me (1989) |  | The Lord of the Rings: The Fellowship of the Ring (2001) |  |

