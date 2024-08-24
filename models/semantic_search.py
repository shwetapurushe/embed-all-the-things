from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# downloaded to ~/.cache

query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district',
                                  'Delhi is the capital of India',
                                  'London has an area of 10 million sq kilometers.'
                                  ])

print("Similarity:", util.semantic_search(query_embedding, passage_embedding))