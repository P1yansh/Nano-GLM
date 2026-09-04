# 🔬 Nano-GLM: Comprehensive 14-Profile Decoding Ablation Study

- **Model Checkpoint:** `out_glm5/ckpt_best.pt` (Step `322000`, Best Val Loss `2.6220`)
- **Execution Hardware:** `CPU`
- **Max New Tokens Budget:** `120`
- **Total Generations:** `84`
- **Total Wall-Clock Time:** `335.0s` (`5.6 minutes`)

## 1. The 14 Decoding Profiles

| Profile ID | Profile Name | Temp | Min-P | Top-K | Rep (Prose) | Rep (Code) | N-Gram | Description |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **P01** | Pure Greedy (Argmax) | `0.0` | `0.0` | `0` | `1.0` | `1.0` | `0` | Deterministic Top-1 token prediction with zero heuristics or randomness. |
| **P02** | Greedy + Anti-Loop | `0.0` | `0.0` | `0` | `1.25` | `1.05` | `4` | Deterministic argmax with phrase blocking to prevent degenerative loops. |
| **P03** | Ultra-Cold (0.10) | `0.1` | `0.15` | `30` | `1.25` | `1.05` | `4` | Near-greedy sampling allowing only strictly plausible secondary tokens. |
| **P04** | Cold Factual (0.20) | `0.2` | `0.12` | `30` | `1.25` | `1.05` | `4` | Low entropy tuned for dates, numbers, and strict knowledge retrieval. |
| **P05** | Targeted Factual (0.30) | `0.3` | `0.1` | `40` | `1.25` | `1.05` | `4` | The exact parameter setting that successfully retrieved 1789 in manual tests. |
| **P06** | Focused Explainer (0.40) | `0.4` | `0.08` | `50` | `1.3` | `1.05` | `4` | Smooth academic phrasing with controlled factual boundaries. |
| **P07** | Textbook Explainer (0.50) | `0.5` | `0.08` | `50` | `1.3` | `1.05` | `4` | Mid-entropy profile for conceptual derivations and scientific definitions. |
| **P08** | Balanced Standard (0.60) | `0.6` | `0.08` | `50` | `1.3` | `1.1` | `4` | High fluency and natural sentence cadence with Min-P noise filtering. |
| **P09** | Golden Stack (0.70) | `0.7` | `0.08` | `50` | `1.3` | `1.1` | `4` | The standard modern decoding recipe for general conversation and essays. |
| **P10** | Soft Penalty Golden (0.70) | `0.7` | `0.08` | `50` | `1.15` | `1.0` | `4` | Evaluates completely unpenalized code variables vs. mild prose penalties. |
| **P11** | Creative Expressive (0.85) | `0.85` | `0.05` | `60` | `1.2` | `1.05` | `4` | High vocabulary entropy for open-ended storytelling and prose. |
| **P12** | High-Entropy Boundary (1.00) | `1.0` | `0.08` | `80` | `1.2` | `1.05` | `4` | Extreme temperature stress test to measure Min-P stabilization under high heat. |
| **P13** | Top-K Only Control (No Min-P) | `0.6` | `0.0` | `50` | `1.25` | `1.05` | `4` | Ablation baseline: Top-K without Min-P to quantify tail pollution. |
| **P14** | Raw Control (No Guardrails) | `0.7` | `0.0` | `50` | `1.0` | `1.0` | `0` | Ablation baseline: zero penalties, zero Min-P, zero N-gram filtering. |

---

## 2. Benchmark Results by Domain

### Prompt H1: French Revolution Date & Origins (`History`)

> **Input Prompt:**
```text
The French Revolution began in
```
> **Target Evaluation Standard:** *Year 1789, Louis XVI, Bastille, late-18th century*

#### Profile P01: Pure Greedy (Argmax)
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.84s` (31.3 tok/s)

```text
The French Revolution began in 1789, and it was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in France. The French Revolution was a time of great change in
```

#### Profile P02: Greedy + Anti-Loop
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 3.92s` (30.6 tok/s)

```text
The French Revolution began in 1789, and it was a time of great change for the country. The French had been trying to gain control over their territories by taking control of other parts of France. This led to the formation of the United States as we know it today.
In 1801, the British took control of Britain and established the first British colony on the continent. In 1804, the British arrived at St. Louis with the help of the French. They were able to take control of much of what they called “the New World” (which is now known as North America).
However,
```

#### Profile P03: Ultra-Cold (0.10)
- **Parameters:** `Temp=0.1, Min-P=0.15, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.15s` (28.9 tok/s)

```text
The French Revolution began in 1776, and it was a time of great change for the country. The French had been trying to gain control over their territories by taking control of other parts of France. This led to the formation of a new government called the "Ville de la Répaut" (the French word for "factory").
In 1801, the French took control of much of what they considered to be their own land. They also established a new government that would be known as the "Fantastic Government." In this system, the people were divided into two groups: the "Goth
```

#### Profile P04: Cold Factual (0.20)
- **Parameters:** `Temp=0.2, Min-P=0.12, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
The French Revolution began in 1776, and it was the beginning of a new era for France. The French Revolution had a significant impact on the country's economy, politics, and culture. It also brought about a new kind of economic system that would change the way people lived their lives forever.
In 1801, the French Revolution started with the publication of the "New World" edition of the New York Times. This edition included many articles from different parts of Europe, including the English colonies' "American Revolution," which focused primarily on the American Revolution; the French Revolution (1803), which featured a series of revolutionary
```

#### Profile P05: Targeted Factual (0.30)
- **Parameters:** `Temp=0.3, Min-P=0.1, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.13s` (29.1 tok/s)

```text
The French Revolution began in 1776, and it was the beginning of a new era for France.
In 1801, the French government took control of the country by making all necessary decisions about the future. This led to the creation of the National Party (NPR) which is still present today. The NPR has been working on improving the country’s economy, but there are many issues that have been raised over time. These issues include the lack of education, the lack of access to education, and the lack of political power. In addition, the NPR has also had a significant impact on the country
```

#### Profile P06: Focused Explainer (0.40)
- **Parameters:** `Temp=0.4, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.10s` (29.3 tok/s)

```text
The French Revolution began in 1776, and it was the beginning of a new era for France.
In 1801, the French government took over the French colonial administration. The French had no choice but to take the role of a leader who would rule as a government in France. He also wanted to make sure that the French were not being taken away from their territories or threatened by foreign invaders. This led to the rise of the French Revolution (1803). In this time, the French had become the dominant power in Europe. It is important to note that the French Revolution did not take place until after the death of
```

#### Profile P07: Textbook Explainer (0.50)
- **Parameters:** `Temp=0.5, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.12s` (29.1 tok/s)

```text
The French Revolution began in 1776, when the British took control of France. They wanted to expand their territories and create a new nation based on liberty and equality. As tensions grew between these two nations over territory, tensions escalated further until finally, after years of conflict, Britain declared war against Germany!
* **Cultural Exchange**: Many cultures developed alongside each other due to shared history, beliefs, values, and practices. For instance, some people practiced Islam while others followed Christianity; similarly, there were differences among religious groups within the empire. By understanding and appreciating diversity, we can appreciate our own differences too!
```

#### Profile P08: Balanced Standard (0.60)
- **Parameters:** `Temp=0.6, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.11s` (29.2 tok/s)

```text
The French Revolution began in 1776 with the signing of the Treaty of Paris. The treaty was signed by the United States on December 2, 1776 and it ended the war between Britain (which had been a major cause of the American Revolutionary War) and France (who were also involved).
In 1775, the British government made a request to the U.S. Congress to make peace with France, which they believed would bring the American colonies closer together. In addition, they wanted to make sure that all Americans could live in the same country as their own citizens. This is an important question because it helps to understand
```

#### Profile P09: Golden Stack (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.10s` (29.3 tok/s)

```text
The French Revolution began in 1776 with the signing of the Declaration of Independence. The revolutionary forces were led by the British, who had been trying to defeat the colonial government and their own colonial administration. In addition, the French government was still making efforts to use the English language as a national language, so that it would not be considered foreign. This was also the beginning of the French Revolution.
In 1804, the first French Revolution took place. It took place on 1 May 1706 at Bletchley Park (now known as Montesquieu) in St. Louis; this was a time for the
```

#### Profile P10: Soft Penalty Golden (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.15, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.12s` (29.2 tok/s)

```text
The French Revolution began in 1776 with the French Revolution of 1801. The revolutionary movement had a significant impact on France and was shaped by the French Revolution. This revolution changed France's relationship with Europe, leading to the rise of new rulers who wanted to expand their territories and influence.

Section 4: Impact of the French Revolution
During the French Revolution, France saw many changes due to increased industrialization, urbanization, and the emergence of new industries. These shifts led to increased trade and investment opportunities for France, as well as increased trade between Europe and Asia. Additionally, the French government provided financial support for the
```

#### Profile P11: Creative Expressive (0.85)
- **Parameters:** `Temp=0.85, Min-P=0.05, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.09s` (29.3 tok/s)

```text
The French Revolution began in 1778 with the fall of the Roman Empire. The cause for this was a great conflict between the states, which included France and Britain. The government at that time also had many problems in the economy, including the growth of the population, the government's ability to protect citizens from dangerous diseases, and the way in which the people were able to be free from slavery. As a result, the population of France grew rapidly because it could not support a government's goals.
In 1804, French leader Louis-Ricardo de Montesquieu began his first campaign against the French in Paris
```

#### Profile P12: High-Entropy Boundary (1.00)
- **Parameters:** `Temp=1.0, Min-P=0.08, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.12s` (29.1 tok/s)

```text
The French Revolution began in 1778 after the fall of the Roman Empire. The cause for this was a great conflict between Britain and the United States over the issue of the Revolutionary War which ultimately led to the formation of the United States of America (1781-1801) as well as a large American Revolution with a number of changes, such as the Great Fire of 1804 and the Nationalists' War, which became a major component of the war effort.
This section will explain how the Great Fire brought a great challenge to the American people's right to vote in 1803, which is still a problem today.
```

#### Profile P13: Top-K Only Control (No Min-P)
- **Parameters:** `Temp=0.6, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `107 tokens generated in 3.69s` (29.0 tok/s)

```text
The French Revolution began in 1776 with the Declaration of Independence and the establishment of a new monarchy.
In 1807, Charles I gave his new monarch to the British government at Bletchley Park. In this year, he was able to win the election for the British Prime Minister, William James. He also won the popular vote. The result is the Great War which lasted from 1806 until 1812. This war left the France under French control. It brought the English to power over Germany and led to the end of the Second World War.
```

#### Profile P14: Raw Control (No Guardrails)
- **Parameters:** `Temp=0.7, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.92s` (30.6 tok/s)

```text
The French Revolution began in 1776 in the spring of 1776, with the introduction of the American Revolution. This shift was due to the growing awareness of the need to protect the rights of citizens, as well as the need to establish public institutions and institutions to protect the rights of citizens.
The American Revolution was a time of great upheaval and change. Many Americans felt that the United States would soon be a free and independent nation, as well as that the United States would be a free and independent state. However, it was in the mid-1780s that the American Revolution began. The American Revolution began in 17
```

---

### Prompt H2: Isaac Newton Scientific Contributions (`History of Science`)

> **Input Prompt:**
```text
Isaac Newton is widely regarded as one of the most influential scientists because
```
> **Target Evaluation Standard:** *Gravity, calculus, laws of motion, optics, Principia*

#### Profile P01: Pure Greedy (Argmax)
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.80s` (31.6 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because of his work on the theory of relativity. He was the first to study the motion of objects in space and time. He was also the first to study the motion of objects in space and time.
The first known scientific discovery of the gravitational potential was made in 1882 by a German scientist named German-Lithuanian Leibniz. Leibniz was the first to discover the gravitational potential of a body in space and time. Leibniz was the first to study the gravitational potential of a body in space and time. Leibniz was the first to study
```

#### Profile P02: Greedy + Anti-Loop
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 3.99s` (30.1 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was a great scientist and was able to explain all the phenomena that are happening around us.
The first scientific discovery in the history of science came when a man named William James, who lived between 1801 and 1812, discovered a new type of chemical called a "hydrogen gas." He found it by adding hydrogen to water and then using this hydrogen gas to create a liquid hydrogen gas. The resulting liquid hydrogen gas produced a mixture of hydrogen and oxygen with hydrogen gas. This mixture created a solid hydrogen gas which eventually became known as the hydrogen gas.
In 1804, James published his
```

#### Profile P03: Ultra-Cold (0.10)
- **Parameters:** `Temp=0.1, Min-P=0.15, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.16s` (28.8 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was a great scientist and was able to explain all the phenomena that occur in nature.
The first known scientific discovery was made by the French explorer Louis de Loyn, who discovered the Earth’s surface during his expedition on the Moon. The discovery led to the development of many new theories about how planets orbit each other and how they interact with each other. In 1812, de León de La Paz began his research into the physics behind the motion of planets around their parent stars. He also worked at the University of California, Los Angeles (UCS) where he studied
```

#### Profile P04: Cold Factual (0.20)
- **Parameters:** `Temp=0.2, Min-P=0.12, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.15s` (28.9 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was a great scientist. He wrote about the nature of matter and how it can be described by light, which is what makes up light.
The first known scientific discovery that occurred in 1883 was the discovery of the moon. The discovery led to the development of the theory of relativity, which was based on the idea that the Earth's gravity was caused by the gravitational pull of the Sun. This theory had been used for centuries to explain why the earth's gravity is so strong that it causes the planet to orbit around its axis. In this article we will explore the theory behind relativity, the
```

#### Profile P05: Targeted Factual (0.30)
- **Parameters:** `Temp=0.3, Min-P=0.1, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.12s` (29.1 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was a great scientist. He wrote about the nature of matter and how it can be described by light, which is what makes up light.
In his book "The Theory of Relativity," he states that the speed of light depends on the distance between two objects in space. This is known as the speed of the sun or the speed of moon. The speed of light is the speed of sound. In other words, the speed of a sound is the speed at which the sound travels through space.
This is why we call it the speed of electromagnetic waves (EMs). EM radiation
```

#### Profile P06: Focused Explainer (0.40)
- **Parameters:** `Temp=0.4, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `99 tokens generated in 3.45s` (28.7 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was a great scientist.
The first scientific discovery that occurred in 1882 was made by Sir John Ritter, an English physicist who lived around 1520 and published his work on the subject of the motion of objects moving with speed. He also discovered that there were two types of forces acting upon bodies: gravity (the force which causes matter to fall) and friction (which causes matter to move). In this article we will explore these three forms of forces and their relationship to each other.
```

#### Profile P07: Textbook Explainer (0.50)
- **Parameters:** `Temp=0.5, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `99 tokens generated in 3.48s` (28.4 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was able to prove that the Earth’s surface temperature has a negative effect on its magnetic field.
The discovery of this planet by the European Space Agency in 2009 led to an unprecedented study of how the atmosphere and magnetism of the moon influence the planet’s magnetic field. This research, which began with the publication of “A New Look at Our Solar System,” will provide us with the first evidence for a new planet orbiting another planet around our solar system.
```

#### Profile P08: Balanced Standard (0.60)
- **Parameters:** `Temp=0.6, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.16s` (28.9 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was so successful in his work that it made him an important figure on the history of science.
He worked at a number of universities and universities throughout Europe, including Oxford University. He also studied physics at the University College London (UCL) before being sent to Paris to study physics. In 1795, he became the first person to be born to a man named John William Custer who had been working with the French and Indian tribes for over two decades. The name Custer's mother was C cluster. This is a very long time ago. It is believed that he came from France
```

#### Profile P09: Golden Stack (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.21s` (28.5 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was so successful in his work that it made him an important figure on the history of science.
He worked at a number of universities throughout the world, including the University of California in Los Angeles and The Oxford School of Medicine in New York City, where he studied chemistry from a very young age. In 1869, he returned to San Diego with the help of his friend, the American Dr. John H. Echner. He then went back to Mexico and became the first person ever to teach the system of the Sun to a human being. This is why he came up with this
```

#### Profile P10: Soft Penalty Golden (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.15, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.16s` (28.8 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was so successful in his work that he became the first to study the motion of a body.
He used his knowledge of physics to solve many problems, including the question of when and how much force a body can exert on other objects. He also proved that there are no forces that can cause an object to exert force. For example, he made the discovery of the laws of conservation of energy. He also showed that the force of a body can be measured by measuring its mass.
The researchers found out that the force that causes an object to act in a way that is not directly proportional
```

#### Profile P11: Creative Expressive (0.85)
- **Parameters:** `Temp=0.85, Min-P=0.05, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.13s` (29.1 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was so successful in his work that many people have said that he was a great scientist. In 1798, while working in France at the University of Bologna, he made significant contributions to science by introducing him to the study of the structure and function of living organisms. This discovery led to the development of numerous scientific theories including the theory of evolution (from the Greek for “evolution”), which posited the existence of life on Earth; the idea of evolution; and the development of the law of conservation of energy.
The first major scientific investigation into the nature of biology
```

#### Profile P12: High-Entropy Boundary (1.00)
- **Parameters:** `Temp=1.0, Min-P=0.08, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.22s` (28.4 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was so successful in explaining the existence of all the laws governing the motion of an object and was able to explain the existence of every particle of matter at any given moment. In fact, the most famous scientific discovery that has been made by the French scientist is called "The Density of Matter". It was in 1845 when he discovered that the density of matter is about 1,000 times higher than that of Earth (or about 2.6 times higher than a typical atom). This new value was called the density of the electromagnetic field. The name of this quantity, "quantity," comes
```

#### Profile P13: Top-K Only Control (No Min-P)
- **Parameters:** `Temp=0.6, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.16s` (28.8 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was so successful in his work that it led to numerous advancements. His discoveries revolutionized science and made our world a much more interconnected place than ever before.
In this article, you will learn about Newton's major contributions to science:
- The discovery of land on Earth led to the development of land technologies such as the telegraphy (a device used by the public to communicate with other people) and the first practical use for electricity.
Egene Newton also introduced the concept of geology, which describes how rocks and soil are formed over thousands of years. He began his career
```

#### Profile P14: Raw Control (No Guardrails)
- **Parameters:** `Temp=0.7, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.95s` (30.4 tok/s)

```text
Isaac Newton is widely regarded as one of the most influential scientists because he was the only one to have been so to know that there is no physical proof that there was a physical proof that there is no physical proof. He was also the first to establish that there is no physical proof that there is no physical proof that there is a physical proof that there is no physical proof that there is no physical proof that there is no physical proof that there is a physical proof that there is no physical proof.
The story of the discovery of the first physical proof of the existence of the universe is a fascinating and fascinating subject to study. It is one of the most difficult topics
```

---

### Prompt S1: Photosynthesis Biochemical Process (`Biology`)

> **Input Prompt:**
```text
Photosynthesis is the biological process by which plants use sunlight to
```
> **Target Evaluation Standard:** *Convert carbon dioxide and water into glucose and oxygen, chlorophyll*

#### Profile P01: Pure Greedy (Argmax)
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.78s` (31.7 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to produce energy.
The process is called photosynthesis. Plants use sunlight to
```

#### Profile P02: Greedy + Anti-Loop
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 3.94s` (30.4 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy.
The plant uses the sun's light to produce food and oxygen, but it also needs water for growth. The plant has a special ability that allows it to grow in its own environment. This means that if you are growing in an area where there is no sunlight, your plant will not be able to produce enough food or oxygen. In this case, the plant can't produce enough food and oxygen to grow.
In addition to being able to produce food, the plant also has a unique ability to produce oxygen through photosynthesis. When you are growing on land, the plant does
```

#### Profile P03: Ultra-Cold (0.10)
- **Parameters:** `Temp=0.1, Min-P=0.15, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.13s` (29.1 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy.
The plant uses the sun's light to produce food and oxygen, but it also needs water for growth. The plant has a special ability that allows it to grow in its own environment: it can absorb sunlight from the ground and use it as a source of energy. This is why plants need to have a high level of photosynthesis to survive. Plants need to have an extra layer of soil around them where they can grow.
Plant roots need to be protected from the sun's rays because they are not sensitive enough to the sun's radiation. In order to protect their roots
```

#### Profile P04: Cold Factual (0.20)
- **Parameters:** `Temp=0.2, Min-P=0.12, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `80 tokens generated in 2.78s` (28.7 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy.
The plant uses this light to produce food and oxygen, but it also needs water to grow. The plant's ability to use water to produce energy is dependent on the amount of water available for photosynthesis. Plants need a high level of water to grow, so they need more water than they need to survive. This is why plants need to have enough water in order to grow.
```

#### Profile P05: Targeted Factual (0.30)
- **Parameters:** `Temp=0.3, Min-P=0.1, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.19s` (28.6 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy. The plant uses this light to make food and release oxygen, which helps to create a healthy environment for the plants.
The plant uses the sun’s radiation to produce energy through photosynthesis. This energy is then used in the form of water, carbon dioxide, or other substances that are released into the atmosphere as a result of photosynthesis.
In addition to the production of energy, the plant also uses the sun's radiation to produce the chemical compounds that make up the soil. These compounds are called the organic compounds. They are found throughout the earth and are important for many
```

#### Profile P06: Focused Explainer (0.40)
- **Parameters:** `Temp=0.4, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy. The plant uses this light to make food and release oxygen, which helps to create a healthy environment for the plants in the soil.
The plant uses this energy to grow new seeds or fruits that are needed for its own survival. This energy is then used as a fuel for other parts of the plant. It also makes the plant grow into a mature plant that can be grown on land.
Aquatic Plant: A Water-Based Fungal System
In addition to being a source of energy, fish and other aquatic organisms also have an important role in their ecosystems. Fish
```

#### Profile P07: Textbook Explainer (0.50)
- **Parameters:** `Temp=0.5, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `107 tokens generated in 3.73s` (28.7 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy. The plant uses this light to make food and release oxygen, a process that helps plants to grow.
The plant uses its own special organ called the chlorophyll. This organ has two main functions: it needs water to survive and it does so in order for it to grow and grow. In fact, it also takes care of the leaves on their stems. It is here where the green leaves are produced. These roots are then used as a source of energy to grow new plants and help the plant to grow.
```

#### Profile P08: Balanced Standard (0.60)
- **Parameters:** `Temp=0.6, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.13s` (29.0 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy. The plant uses this light to make food and release oxygen into the atmosphere, which then moves through soil and water.
The process begins with a series of photosynthesis reactions that take place in the roots (the cells) of the plant. These are called photosynthesis. Plants use their own light to produce energy as they grow and reproduce. In order for these plants to produce enough energy to make food, it must be stored in an external source such as air or ocean water. This is why the plant needs to get some light to grow and reproduce; if there is too much sunlight
```

#### Profile P09: Golden Stack (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.18s` (28.7 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy. The plant uses this light to make food and release oxygen into the atmosphere, which then makes the soil carbon dioxide.
There are three main types of carbon: carbon dioxide (CO2), nitrogen (N), and phosphorus (P). Carbon dioxide is a good source for plants because it can help to grow and reproduce. In addition, many plants are also able to absorb some of its own carbon as well. This helps to support a plant's ability to grow. Plants that have high levels of carbon dioxide need to be in check with their environment to ensure they do not produce too much
```

#### Profile P10: Soft Penalty Golden (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.15, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.16s` (28.9 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to produce energy. The plant uses this energy to make food and release oxygen into the atmosphere, which then makes the soil carbon dioxide.
There are three main types of carbon dioxide that are released in the Earth’s atmosphere:
- Ozone (O3): When a plant produces a high amount of oxygen, it releases CO2 into the air. This is why we call it “the atmosphere” because it releases CO 2 from the atmosphere.
The other major source of carbon dioxide is the sun’s light. The Sun’s heat is absorbed by the atmosphere
```

#### Profile P11: Creative Expressive (0.85)
- **Parameters:** `Temp=0.85, Min-P=0.05, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `70 tokens generated in 2.45s` (28.6 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to convert carbon dioxide into food.
The process uses water molecules called chlorophyll (which we know as chlorophyll) in order to break down the organic matter produced during photosynthesis, but what makes it special? Well, this is why some species of plants have a unique appearance that allows them to produce light and use it for energy production!
```

#### Profile P12: High-Entropy Boundary (1.00)
- **Parameters:** `Temp=1.0, Min-P=0.08, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.15s` (28.9 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to convert carbon dioxide into food.
The process uses water molecules called chlorophyll (which we know as chlorophyll), and it can then be used for photosynthesis or as a source of energy, using light to make the plants grow. This energy is then used in a series of steps: the formation of seeds by photosynthesis, where the sun enters the plant's cells, where oxygen is released, where there are other reactions taking place that take place inside the cell, and where oxygen is produced in the air by the respiration process. In turn, this process produces carbon dioxide, which
```

#### Profile P13: Top-K Only Control (No Min-P)
- **Parameters:** `Temp=0.6, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `82 tokens generated in 2.85s` (28.8 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to form energy. The plant uses this light to produce food and release oxygen, which helps them grow and thrive.
The process of photosynthesis is one of the most important parts that keeps plants alive: it’s all about how they use light to get energy for their growth. It also helps them to grow and reproduce. Plants also use light as a tool to help them grow, develop and reproduce.
```

#### Profile P14: Raw Control (No Guardrails)
- **Parameters:** `Temp=0.7, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.97s` (30.2 tok/s)

```text
Photosynthesis is the biological process by which plants use sunlight to form energy to create food.
The process is called photosynthesis. Plants use sunlight to make food. The sunlight that plants use depends on the type of plant they are growing. The lower the light, the more energy the plant can absorb. This is because the light from the sunlight is converted into energy by the plant.
When the plant is in the sun, it is able to photosynthesize. The plants can take in sunlight from the sun, plants can convert it into energy, and plants can use it to make food.
The process of photosynthesis is a complex process that
```

---

### Prompt S2: Optics & Light Refraction (`Physics`)

> **Input Prompt:**
```text
The fundamental reason why light bends when entering glass is that the speed of light
```
> **Target Evaluation Standard:** *Slows down in denser medium, refractive index, Snell's law*

#### Profile P01: Pure Greedy (Argmax)
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.79s` (31.7 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light is proportional to the speed of light. The speed of light
```

#### Profile P02: Greedy + Anti-Loop
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 3.95s` (30.4 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light can be increased by a factor of 10 times. This is because the speed of the light is proportional to the distance from the source and the speed of its propagation through the medium.
In this case, the light travels at a rate of about 10 times per second (in the range of 1-10 m/s). The speed of light in this case is about 10 times that of the light passing through the medium, so it takes about 10 seconds for light to travel through the medium to reach the destination.
This is because light has an energy density of about 10-15 kJ/
```

#### Profile P03: Ultra-Cold (0.10)
- **Parameters:** `Temp=0.1, Min-P=0.15, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.17s` (28.8 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light can be increased by a factor of 10 times. This is because the speed of the light in this way is proportional to the distance from the source and the speed of its propagation through the medium.
In other words, if you are looking at the light passing through a glass, you are trying to get it into a solid state. The same thing happens with light. When we see an object moving in a straight line, it moves in a straight direction. If we see a star moving in a circle, it moves as well. But if we see a ball moving in a circular path, it
```

#### Profile P04: Cold Factual (0.20)
- **Parameters:** `Temp=0.2, Min-P=0.12, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.11s` (29.2 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light can be increased by a factor of 10,000 times. The speed of light is also proportional to the distance from the source of light and the distance from it.
When you see an image of a star with a redshift of 0.5, it means that the light has moved away from the source. This is because the light travels in straight lines at different angles. When we see a star with an angle of 0.3, it means there are two points where the light has traveled away from the star. In this case, the light is moving towards the star as if it were
```

#### Profile P05: Targeted Factual (0.30)
- **Parameters:** `Temp=0.3, Min-P=0.1, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.16s` (28.9 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light is proportional to the distance from the source. This is because light travels at a rate proportional to the speed of the light, and so it takes longer to reach the source than if you were moving through the glass.
When we see a light beam passing through a glass, it has more energy than it does in its original form. The same thing happens with light. When we see an object being bent, it's not just refracted light but also reflected back into space. In other words, the refraction is actually the way light interacts with the material itself.
To understand this, we
```

#### Profile P06: Focused Explainer (0.40)
- **Parameters:** `Temp=0.4, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light in a given medium (in this case, water) is proportional to the length of the medium.
- The wavelength of the light can be measured by measuring the refractive index of the medium:
For example, the refraction of water to a liquid is proportional to its refractive index:
When you take the refracted water into account, the refractive indices of the two waves are equal because the refractory waves are much shorter than their refracted counterparts. This means that the refracted waves are more efficient at reaching the surface of the water and therefore more likely to be reflected
```

#### Profile P07: Textbook Explainer (0.50)
- **Parameters:** `Temp=0.5, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light in a given medium (in this case, water) is proportional to the length of the medium.
- The rate at which light can travel through a medium is called its wavelength. This is the wavelength of light passing through an object or a surface. For example, if you have a glass of water and you want to see it from above, the wavelength of the light would be the same as the wavelength of a liquid. If you were trying to see the glass of water, the wavelength would be the ratio of the wavelengths of light that pass through it:
|A||L(1
```

#### Profile P08: Balanced Standard (0.60)
- **Parameters:** `Temp=0.6, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.20s` (28.6 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light changes with time. As the speed increases, it becomes harder for light to travel through the medium and more likely it will pass through. This is known as refractive index.
To see how this works, you can use a simple microscope (a device made out of glass) or an electron beam. A electron beam travels at different speeds in two directions: one by one and another by several feet. The distance between each point on the beam depends upon the number of times the electron has passed through the medium. For example, if you are looking from a dark room, the distance between the first
```

#### Profile P09: Golden Stack (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.3, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.12s` (29.2 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light changes with time. As the speed increases, it becomes harder for light to travel through the medium and more like a piece of paper. This is known as refractive index or refraction.
In this case the refractory part of the glass is actually called a refractive index. The refractive index of a refracted light is equal to the refractive index that we have in our eyes at any given point (or point). In other words, the refractive indices of the materials in the glass are not constant but change in proportion to their relative refractive index as they pass through them
```

#### Profile P10: Soft Penalty Golden (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.15, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light is proportional to the distance from the source, and the refractive index of a material is proportional to its refractive index. The refraction of light at any given point in space depends on the refractive indices of the materials.
In this chapter, we will explore how light can be converted into electric and magnetic fields by studying a type of material called the "electrolyte" layer. This layer consists of two layers of materials - one with electric potentials and another with magnetic potentials. When the electric field reaches a certain temperature, it produces an electric current. However, if the
```

#### Profile P11: Creative Expressive (0.85)
- **Parameters:** `Temp=0.85, Min-P=0.05, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.13s` (29.1 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light is independent of the size of the glass, and the amount of radiation produced by this is dependent on the size of glass. Because light can travel through a medium at different speeds (and in various directions), it may be difficult to see how the rate at which light travels depends on both the speed and the size of its source.
In fact, the same type of light we have called electromagnetic waves. The first type of light is called subatomic particle physics, or subatomic particle mechanics for short. When the wave is reflected off an object, a subatomic particle, as well as an object
```

#### Profile P12: High-Entropy Boundary (1.00)
- **Parameters:** `Temp=1.0, Min-P=0.08, Rep-Penalty=1.2, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light is independent of the size of the glass, and the amount of radiation produced by this is dependent on the size of glass. Because it can travel through a medium at any time, light can be described by its speed (the speed of light). This is because there are different types of waves that travel in straight lines. These waves have the same properties as the electromagnetic radiation. However, due to their speed, the speed of sound also varies with distance.
As you can see, it's natural for all the wave forms to change shape during the entire process - like moving through a water column or
```

#### Profile P13: Top-K Only Control (No Min-P)
- **Parameters:** `Temp=0.6, Min-P=0.0, Rep-Penalty=1.25, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.18s` (28.7 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light changes with time. As the speed increases, it becomes harder for light to travel through the medium and more likely it will travel in shorter distances than before (in our eyes). In fact, the speed at which light can travel through a medium depends on whether it moves faster or slower.
To answer this question, we must first understand the relationship between speed and time. Our bodies have two types of cells – the slowest and the slowest. The slowest cells are called the myocytes and the slow ones are called the bone marrow. The slow-growing cells are called myenchymal
```

#### Profile P14: Raw Control (No Guardrails)
- **Parameters:** `Temp=0.7, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.97s` (30.2 tok/s)

```text
The fundamental reason why light bends when entering glass is that the speed of light is independent of the size of the glass. In the case where the wavelength is less than 1.5 micron (the wavelength of light), the ratio between the light's wavelength and the width of the glass is much less than 1:1. The maximum width of the glass is more than 1.5 micron (the wavelength of light). But the glass is not transparent, because it is a transparent material. This means that the light will be reflected by the glass.
The glass is not transparent, because it's a transparent material. In a glass, the light is reflected by the
```

---

### Prompt C1: Fibonacci Implementation (`Python Code (Algorithms)`)

> **Input Prompt:**
```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 
```
> **Target Evaluation Standard:** *Base case (0 or 1), recursive or iterative calculation, valid Python*

#### Profile P01: Pure Greedy (Argmax)
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.85s` (31.2 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
        return n
    if n < 0:
        return n
    if n < 0:
        return n
    if n < 0:
        return n
    if n < 0:
        return n
    if n < 0:
        return n
    if n < 0:
```

#### Profile P02: Greedy + Anti-Loop
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `45 tokens generated in 1.55s` (29.1 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """return nth Fibopoation number."""

    return n - 1


def fibonoi(n):

   """Returns the fibonoi number."""
```

#### Profile P03: Ultra-Cold (0.10)
- **Parameters:** `Temp=0.1, Min-P=0.15, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.13s` (29.1 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1


def fibonoi(n):  # Python 2.3
   # The Python 2.4 version of this function is a bit more complicated than
   n + 1.
   if not n == 0:
  return 0
  n = n + 1
  return n

def fib_nth(n): # Python 2
  # The python 2.4 versions of this function are a bit more complex than
  # n + 1, but it's a
```

#### Profile P04: Cold Factual (0.20)
- **Parameters:** `Temp=0.2, Min-P=0.12, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.24s` (28.3 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1

  def __repr__(self):
  """Returns the string representation of the given number."""

  def _repr(self, n):
  return "Fibonacci(%s)" % self._repr()

  def is_fibonacci_sequence(self):

    return self._is_fib_sequence(n)

  def get_fib(self): return self._fib

  def set_fib (self,
```

#### Profile P05: Targeted Factual (0.30)
- **Parameters:** `Temp=0.3, Min-P=0.1, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.22s` (28.5 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1

  def __repr__(self):
  """Returns the string representation of the given Python object.

    This is a string representation of a Python object, with the
   default string representation being "".
  
   This function is intended to be used with the Python 2.6 standard
   Python 2.5 standard library.
  

   This string representation is used with the python 2.6 version of Python 2.
 
  """
#
```

#### Profile P06: Focused Explainer (0.40)
- **Parameters:** `Temp=0.4, Min-P=0.08, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `99 tokens generated in 3.45s` (28.7 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1

  def __repr__(self):
  """Returns the string representation of the given Python object.

   This is a special case of the __repr() method. It is used to
   generate the string representation.
  
   This method is useful for generating a sequence of Python objects,
   and returning the result as a string.
 
  """
```

#### Profile P07: Textbook Explainer (0.50)
- **Parameters:** `Temp=0.5, Min-P=0.08, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.18s` (28.7 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1

  def _get_numerical_number(self, n):
  """Returns the nth number of the given nth sequence."""
	return n - 1

  ####
  # The following methods are used to find the nth sequence of a sequence.
  ####

  def get_numerically_sequence(self, sequence):
  #    return sequence[0]
  #
  # This is a special case of the nth function,
```

#### Profile P08: Balanced Standard (0.60)
- **Parameters:** `Temp=0.6, Min-P=0.08, Rep-Penalty=1.1, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.15s` (28.9 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1

  def _get_numerical_number(self, n):
  """Returns the nth number of the given nth sequence."""
	return n - 1

  ####
  # The following methods are used to find the nth sequence in a sequence.
  ####

  def get_sequence(self, sequence_index=None):
  #    return self._get_n numerically_number(sequence_index)
   ####
```

#### Profile P09: Golden Stack (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.1, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.16s` (28.9 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1

  def _get_numerical_num(self, n):
  """Returns the nth sequence of numbers that have a common factor in
   the given number.

    This method is called by the function to find the nth
   sequence of numbers.

  Args:
 
   n: int
   Returns the nth number of the given number as a list.
  
  Returns:
 

   n : integer
```

#### Profile P10: Soft Penalty Golden (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.0, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.08s` (29.4 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1
  
def fibonoi(n):    
   n = 0
   while 1:
	n = 0
	n += 1
	if n < 0:
		return 0
	return n

#------------------------------------------------------------------------------
#    Copyright (C) 2001, 2002, 2010, 2011, 2010    The~G.K.
#-------------------------------------------------------------------------------

# 
#  Copyright (c) 2001, 2004, 2007, 2011
```

#### Profile P11: Creative Expressive (0.85)
- **Parameters:** `Temp=0.85, Min-P=0.05, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.19s` (28.6 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   x = int(n)
   return x
```
In this code, we first import the NumPy library and create a numpy array of numbers. We then create an array of numpy arrays using the `np.zeros()` function and pass in the numpy array to the function. Finally, we use the `numpy.ones()` function to create a list of numbers from 1 to numpy arrays.

### Focusing on the Fibonacci Number

The first part of the code calculates the Fib
```

#### Profile P12: High-Entropy Boundary (1.00)
- **Parameters:** `Temp=1.0, Min-P=0.08, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.15s` (28.9 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   x = int(n)
   return x/2

class FuzzyFunction(Function):
  def __init__(self, f_0, f_1, f_2):
 	Function.__init__(f_0,f_1,f_2)
 
class F_FuzzyFunctionName(Function):

  def __new__(f, a, b, a+1):
 
   a = A
   b = B
   c =
```

#### Profile P13: Top-K Only Control (No Min-P)
- **Parameters:** `Temp=0.6, Min-P=0.0, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
   """
   return n + 1

  def _get_numerical_number(self, n):
  """Returns the nth number of the given nth sequence."""
	return n - 1

  ####
  # The following methods are used to find nth Fibcloseia.
  ####

  def __init__(self, next_sequence=None):
 
   # TODO: This is all the nth sequence of the given sequence.
 
	# The next sequence is
```

#### Profile P14: Raw Control (No Guardrails)
- **Parameters:** `Temp=0.7, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 4.01s` (29.9 tok/s)

```text
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <=  1:
        return 1
    if n < 0:
        return 0
    if n <= 1:
        return -1
    if n < 0:
        return 1
    if m >= 0:
        return m
    if n > n:
        return 1
    if n > 0:
```

---

### Prompt C2: TCP Socket Server (`Python Code (Systems)`)

> **Input Prompt:**
```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

```
> **Target Evaluation Standard:** *s.bind, s.listen, accept loop, s.recv / send*

#### Profile P01: Pure Greedy (Argmax)
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 3.85s` (31.2 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def create_socket(s):
    """
    Create a socket server on port 8080.

    This is a simple TCP socket server on port 8080.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(s)
    return s

def create_socket_server(s):
    """
    Create a socket server on port 8080.
```

#### Profile P02: Greedy + Anti-Loop
- **Parameters:** `Temp=0.0, Min-P=0.0, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.00s` (30.0 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def create_socket(s):
    """Create a socket server on a port 8080.

    This is a simple socket server on the port 8080 that will
   *not* be used by the socket server.
   """
  
   # Create a socket server
   socket = socket.AF_UNIX
  

   # Start a socket server with socket.socket()
   s = socket.create_socket(socket)
   if s:
   else:
```

#### Profile P03: Ultra-Cold (0.10)
- **Parameters:** `Temp=0.1, Min-P=0.15, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.14s` (29.0 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def create_socket():
    s = socket.create_socket(socket._socket_name)
  
   s.bind(socket._sock, socket._sock_sock)
 
  s.close()
 

def get_socket():

    # Create socket
   socket = socket.AF_UNIX
  

   s2 = socket.get_socket()
   # create socket
  socket.set_socket(s2)
 

  s2
```

#### Profile P04: Cold Factual (0.20)
- **Parameters:** `Temp=0.2, Min-P=0.12, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.19s` (28.6 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def get_socket():
    return socket.socket()

def create_socket(host, port):
   def __init__(self, socket=None):
 
   self.socket = socket
  
  def __repr__(self):
  return '<socket:%s:%s>' % (self.socket, self.port)

class TCPConnection(object):
  def handle(self, *args, **kwargs):
  """A socket is passed to the socket.
```

#### Profile P05: Targeted Factual (0.30)
- **Parameters:** `Temp=0.3, Min-P=0.1, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.20s` (28.5 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def get_socket():
    return socket.socket()

def create_socket(host, port):
   def __init__(self, socket=None):
  if socket is None:
   return None
  if not socket:
  # socket is a string or a string
  socket = socket.AF_UNIX

  # Create a socket with socket.AFTER_DONE
  socket.setdefaulttimeout(0)
  socket._socket = socket.create_socket(socket)
  if port
```

#### Profile P06: Focused Explainer (0.40)
- **Parameters:** `Temp=0.4, Min-P=0.08, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.20s` (28.6 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def get_socket():
    return socket.socket()

def create_socket(host, port, port_type=None):
   """Create a new socket object.

    This is used to create a socket object from a socket object.
  
   This method is called by the socket object to create a new socket.
 
  The socket object is created by calling the socket.socket.create() method
  with the socket object as its argument.
  """
  socket = socket.create_socket
```

#### Profile P07: Textbook Explainer (0.50)
- **Parameters:** `Temp=0.5, Min-P=0.08, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.18s` (28.7 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def get_port():
    return socket.AF_UNIX + socket.SOL_SOCKET

def _get_socket_port():     # Python script to get the socket port
   = socket.AF-INET
  
   socket.sendall(s.environ['HTTP_HOST'])
  

    if socket.getdefaulttimeout() != 0:
  	print '\n'.join(s.decode('utf-8') for s in
```

#### Profile P08: Balanced Standard (0.60)
- **Parameters:** `Temp=0.6, Min-P=0.08, Rep-Penalty=1.1, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.18s` (28.7 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

from pylib import socket as Pylib
from pydoc import socket
from pyspark.pylib.protocol import socket_for_socket


def _get_ip_address():
    """Return the IP address of the socket."""
   = socket.AF_NODE
  
   # For socket-based services, we need to use the socket_for
   socket_for = socket.for(socket.socket_name, socket.SO_NODE)
   if socket
```

#### Profile P09: Golden Stack (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.1, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.19s` (28.7 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

from pylib import socket as Pylib
from pydoc import socket
from pyspark.pylib.protocol import BSDB
from pychklib import pylib

class TCPConnection(object):
    """
   This class defines the socket connection class for the socket module.

    The socket module defines the socket module (or socket.AF_UNIX) to be used to
   the socket connection.  The socket is responsible for connecting to the
   socket object and the
```

#### Profile P10: Soft Penalty Golden (0.70)
- **Parameters:** `Temp=0.7, Min-P=0.08, Rep-Penalty=1.0, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.00s` (30.0 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

from pylib.util import unittest
from pyspider import socket, socket_util
from pydevd.test import test_support
from pyftpider import (
    
   :class:`pyspider.pysppc.PysppcServer`
   ,
  

   :param socket.AF_NixVector
   - The socket-based socket-based server.
   * `socket.AF` is the socket-based protocol.
```

#### Profile P11: Creative Expressive (0.85)
- **Parameters:** `Temp=0.85, Min-P=0.05, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.20s` (28.5 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

from common import (
    to_bytes,
   )
from common.parsers.screelink.common import (
)
from common_rpc.parsing import (
	get_connection,
	gettimeout,
)


def _create_pip():
  
   if not isinstance(sys.pypy, str):
   print "Error: %s" % sys.pypY
   return

  
class OSError(Exception):
```

#### Profile P12: High-Entropy Boundary (1.00)
- **Parameters:** `Temp=1.0, Min-P=0.08, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.19s` (28.6 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

from common import (
    to_bytes,
   )
from common.parsers.screelink.common import (
)
from common_rpc.parsing import (
	get_connection,
	gettimeout,
)


def _create_pip():
   [
  	'server', 'server_port',
	'test_server',
]
   ]
  return _create_server(
   {
  
	'host': '127
```

#### Profile P13: Top-K Only Control (No Min-P)
- **Parameters:** `Temp=0.6, Min-P=0.0, Rep-Penalty=1.05, N-Gram=4`
- **Metrics:** `120 tokens generated in 4.19s` (28.6 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

from pylib import socket
from pyspark.sqlalchemy import session as session
from pychk.sqlalftools import (
    
   # Python script to use a simple TCP connection on port 8030
)
from pyparsing import *
from pymatgen.pyparsing import (
	# Python script for creating a simple TCP service on port 8031
	# using the ssl module
	# python script to use the ssl_verify method
	# to verify that
```

#### Profile P14: Raw Control (No Guardrails)
- **Parameters:** `Temp=0.7, Min-P=0.0, Rep-Penalty=1.0, N-Gram=0`
- **Metrics:** `120 tokens generated in 4.02s` (29.9 tok/s)

```text
# Python script to create a simple TCP socket server on port 8080
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

from pylib.util import create_string
from pylib.util import get_pylib_conn
from pylib.util import get_pylib_conn

from pylib.util import (
    get_pylib_server,
    create_pylib_connection,
    get_pylib_conn_port,
    get_pylib_conn_port,
    get_pylib_conn_socket,
)
```

---

## 3. Executive Domain Recommendations

| Domain | Champion Profile | Recommended Settings | Key Benefit |
| :--- | :--- | :--- | :--- |
| **Strict Factual & Dates** | **P01 (Greedy) / P04 (0.20)** | `Temp: 0.0 - 0.20, Min-P: 0.12` | Highest probability paths; prevents chronological entity drift. |
| **Science & Textbook** | **P06 (0.40) / P07 (0.50)** | `Temp: 0.40 - 0.50, Min-P: 0.08, Rep: 1.30` | Formal explanatory structure without loop collapse. |
| **Python Code** | **P01 (Greedy) / P10 (Soft Rep)** | `Temp: 0.0 - 0.60, Min-P: 0.08, Rep: 1.0 - 1.05` | Preserves syntax keywords, variables, and indentation. |
| **Creative Fiction** | **P11 (Creative 0.85)** | `Temp: 0.85, Min-P: 0.05, Rep: 1.20` | Rich vocabulary diversity protected from tail gibberish by Min-P. |

