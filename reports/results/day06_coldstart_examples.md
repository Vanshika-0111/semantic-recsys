# Day 6 — cold-start examples (bge-small)

Pool = most-liked movies in the all-time before onboarding; ranked by cos(profile, movie) + λ·prior. Settings per onboarding type: `{"seeds_1": {"pool": 20, "lambda": 2.0}, "seeds_3": {"pool": 20, "lambda": 5.0}, "seeds_5": {"pool": 20, "lambda": 5.0}, "text": {"pool": 20, "lambda": 1000.0}, "text+seeds_1": {"pool": 20, "lambda": 5.0}, "text+seeds_3": {"pool": 20, "lambda": 5.0}, "text+seeds_5": {"pool": 20, "lambda": 5.0}}`

## Free-text onboarding — nothing but a sentence

'Now' = the end of the data (2023-10-26), so the pool is recent popular movies.

**“a mind-bending science fiction film about dreams and memory”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

**“heartwarming animated movie about friendship for the whole family”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

**“slow, sad drama about grief and losing a parent”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

**“gritty crime thriller with a heist that goes wrong”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

**“feel-good romantic comedy set in New York”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

**“scary folk horror in an isolated village”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

**“epic war movie based on a true story”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

**“funny superhero movie that doesn't take itself seriously”** → The Shawshank Redemption (1994); Pulp Fiction (1994); The Matrix (1999); Forrest Gump (1994); The Silence of the Lambs (1991); Star Wars: Episode IV - A New Hope (1977); Fight Club (1999); Schindler's List (1993); Star Wars: Episode V - The Empire Strikes Back (1980); The Lord of the Rings: The Fellowship of the Ring (2001)

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

