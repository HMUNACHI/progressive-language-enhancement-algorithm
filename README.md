# PLEA: Progressive Language Enhancement Algorithm

![Alt text](/images/plea.png "Plea Diagram")

# AUTHORS
Henry Ndubuaku\
henry@medullaai.com

# ABSTRACT
Language enhancement in this premise means transforming a bland body of text to its rich and nuanced paraphrase. Few-shot fine-tuning of large language models has become the panacea to all sequence-to-sequence problems. Such techniques are however complicated by dataset acquisition and training intricacies. Also, there is little control over the output during inference and  subtle differences in the input and output training pairs are difficult for accessible language models to learn. To this end, this work introduces PLEA, an algorithm that employs controllable computational linguistics to find positions in the input text to transform progressively, and a frozen mask-filling transformer to find the most appropriate replacements or insertions for each.


# BACKGROUND
One of BERT's training pre-training process includes mask-filling (Delvin et. al, 2019)[1]. This invlolves randomly removing tokens from the input sentence and training the transformer to replace it, hence the model learns the most appropriate word to complete the sentence. Most inudtry applications of BERT focus on fewer shot transfer learning whereby the model (or some of its layers) is fine-tuned on relevant datasets. However, works like (Wu et. al, 2019)[2], (Zhang et. al 2021)[3] and (Luitel et. al, 2023)[4] demonstrate BERT's ability to find appropriate words to complete a sentence.

Language enhancement is different from grammar correction, it involves subtle changes in the input sentence to embellish its grammar. Consider the an input text like "Today's sales is better than before, numbers were very bad yesterday." The sample sentence above is both synctatically and semantically correct. However, simply inserting the adjective "much" before "better" transforms the sentence to "Today's sales is much better than before, numbers were very bad yesterday." This strengthens the message it conveys. Large language models discriminates such subtleties as demonstrated by (Wei et. al, 2022)[5]. However, minimal differences in input and output sequences are very difficult for accesible sequence-to-sequence neural networks to detect. To this end, this work proposes PLEA (Progressive Language Enhancement Algorithm).


# APPROACH
PLEA searches for words with specific fine-grained part-of-speech tags, then either inserts a supporting adjectvie/adverb before/after it or replaces the token. This is performed causally, while at each step optimizing for high sentence similarity between the input and the output.

Rule-based comptational linguistics often encounter many compicating edge-cases, hence the lingustic rules were limited to the 4 below with minimal edge cases.

| TAG | NAME                  | EXAMPLE                    | ACTION              | EXAMPLE                            |
| --- | --------------------  | -------------------------- | ------------------- | ---------------------------------- |
| JJR | Comparative Adjective | this is 'better'           | insert mask before  | this is 'significantly better'.    |
| PDT | Predeterminer         | this is 'half' the journey | insert mask before  | this is 'only half' the journey.   |
| RBR | Comparative Adverb    | she slept 'longer'         | insert mask before  | she slept 'relatively longer'      |
| MD  | Modal Auxillary Verb  | I 'can' call you           | replace mask        | I 'could' call you                 |


**PLEA**

```
    let sentence: input sentence
    let top-k: maximum number of possible sentences at each point 
    let s: sentence embedding
    let E: embeddings of potential sentences
    let c: confident probability threshold
    let t: minimum sentencence similarity score
    
    For word in sentence:
    
       if POS-Tag of word is JJR or PDT or RBR:
           Insert mask token before word
    
       if POS-Tag of word is MD:
           replace word with mask token
         
       else:
          continue
          
       chose top-k mask replacements where p(Word_n | Word_0 ... Word_n-1 & Word_n+1 ... Word_end) >= c
       
   return sentences where (s.E / ||s||x||E||) >= t
```


# USAGE
1. Install requirements by running "pip install -r requirements.txt in the command line.
2. Import plea as a library with "from plea import PLEA"
3. Use to enhance a text by creating an instance "plea = PLEA()"
4. Then call the enhance_text with "output_texts = plea.enhance_text(input_text)"


# REFERENCES
1. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Association for Computational Linguistics, Minneapolis, Minnesota, 4171–4186. https://doi.org/10.18653/v1/N19-1423.

2. Wu X, Zhang T, Zang L, et al. “Mask and infill”: Applying masked language model to sentiment transfer. ArXiv: 1908.08039. https://arxiv.org/abs/1908.08039

3. Tianyi Zhang, Tatsunori B. Hashimoto. On the Inductive Bias of Masked Language Modeling: From Statistical to Syntactic Dependencies. ArXiv: 2104.05694. https://arxiv.org/abs/2104.05694

4. Dipeeka Luitel, Shabnam Hassani, Mehrdad Sabetzadeh. Using Language Models for Enhancing the Completeness of Natural-language Requirements. ArXiv: 2302.04792. https://arxiv.org/abs/2302.04792

5. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022. URL https://arxiv.org/pdf/2201.11903.
