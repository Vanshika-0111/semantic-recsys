# Day 4 — Most similar movies (content only, no ratings)

Top-5 neighbours of well-known movies. TF-IDF = Day 3 lexical cosine; the others are FAISS Flat search over sentence-embedding vectors of the SAME item documents.

### Toy Story (1995)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | Toy Story 2 (1999) | Toy Story 2 (1999) | Toy Story 2 (1999) |
| 2 | Toy Story 4 (2019) | Toy Story 4 (2019) | Toy Story 3 (2010) |
| 3 | Toy Story 3 (2010) | Toy Story 3 (2010) | Toy Story 4 (2019) |
| 4 | Buzz Lightyear of Star Command: The Adventure Begins (2000) | Toys (1992) | Toy Story of Terror (2013) |
| 5 | Toy Story Toons: Hawaiian Vacation (2011) | Toy Story Toons: Hawaiian Vacation (2011) | Toys (1992) |

### The Matrix (1999)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | The Matrix Reloaded (2003) | The Matrix Revolutions (2003) | The Matrix Reloaded (2003) |
| 2 | The Matrix Revolutions (2003) | The Matrix Reloaded (2003) | The Matrix Revolutions (2003) |
| 3 | The Matrix Resurrections (2021) | The Matrix Resurrections (2021) | The Matrix Resurrections (2021) |
| 4 | RoboCop (1987) | The Animatrix (2003) | The Terminator (1984) |
| 5 | JUNG_E (2023) | The Machine (2013) | The Animatrix (2003) |

### Titanic (1997)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | A Night to Remember (1958) | Chambermaid on the Titanic, The (Femme de chambre du Titanic, La) (1998) | Titanic (1953) |
| 2 | Titanica (1992) | Titanica (1992) | Titanica (1992) |
| 3 | Titanic (1953) | A Night to Remember (1958) | A Night to Remember (1958) |
| 4 | Raise the Titanic (1980) | Titanic (1953) | Love Affair (1939) |
| 5 | Love Affair (1939) | The Unsinkable Molly Brown (1964) | Chambermaid on the Titanic, The (Femme de chambre du Titanic, La) (1998) |

### Spirited Away (Sen to Chihiro no kamikakushi) (2001)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | Into the Forest of Fireflies' Light (2011) | Spirited (2022) | Origin: Spirits of the Past (Gin-iro no kami no Agito) (2006) |
| 2 | Twitches (2005) | Howl's Moving Castle (Hauru no ugoku shiro) (2004) | Children Who Chase Lost Voices from Deep Below (Hoshi o ou kodomo) (Journey to Agartha) (2011) |
| 3 | Don't Knock Twice (2017) | Into the Forest of Fireflies' Light (2011) | Spirited (2022) |
| 4 | The Last Mimzy (2007) | Tokyo Twilight (Tôkyô boshoku) (1957) | Into the Forest of Fireflies' Light (2011) |
| 5 | Mary and the Witch's Flower (2017) | Disappearance of Haruhi Suzumiya, The (Suzumiya Haruhi no shôshitsu) (2010) | Hello World (2019) |

### The Dark Knight (2008)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | The Dark Knight Rises (2012) | The Dark Knight Rises (2012) | The Dark Knight Rises (2012) |
| 2 | Batman: The Long Halloween, Part One (2021) | Batman (1989) | Batman Begins (2005) |
| 3 | Batman vs. Two-Face (2017) | Batman: The Dark Knight Returns (2013) | Batman (1989) |
| 4 | The Batman (2022) | Joker (2019) | The Batman (2022) |
| 5 | Batman Forever (1995) | The Batman (2022) | Batman: Mask of the Phantasm (1993) |

### The Notebook (2004)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | Woman In Berlin, A (Anonyma - Eine Frau in Berlin) (2008) | Paper Heart (2009) | The Romantics (2010) |
| 2 | Century (1993) | Endless Love (2014) | Amour (2012) |
| 3 | 2046 (2004) | The Romantics (2010) | Iris (2001) |
| 4 | Valentine's Day (2010) | Closer (2004) | Stanley & Iris (1990) |
| 5 | Il Mare (Siworae) (2000) | The Lovers (2017) | The Last Letter From Your Lover (2021) |

### Alien (1979)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | Aliens (1986) | Horror Planet (a.k.a. Inseminoid) (1981) | Aliens (1986) |
| 2 | Alien: Covenant (2017) | Alien Contamination (1980) | Alien: Resurrection (1997) |
| 3 | Cosmos (2019) | Parasite (1982) | Galaxy of Terror (Quest) (1981) |
| 4 | Critters 4 (1991) | It! The Terror from Beyond Space (1958) | The Thing (1982) |
| 5 | Star Trek: Insurrection (1998) | Galaxy of Terror (Quest) (1981) | Horror Planet (a.k.a. Inseminoid) (1981) |

### Inception (2010)

| # | tfidf | minilm | bge-small |
|---:|---|---|---|
| 1 | Cypher (2002) | Criminal (2016) | The Score (2001) |
| 2 | Journey 2: The Mysterious Island (2012) | Thoughtcrimes (2003) | Criminal (2016) |
| 3 | The Cell (2000) | In Dreams (1999) | Dream Man (1995) |
| 4 | Freud: The Secret Passion (1962) | Extracted (2012) | Lucid Dream (2017) |
| 5 | Repeaters (2010) | The Anomaly (2014) | Trance (2013) |

