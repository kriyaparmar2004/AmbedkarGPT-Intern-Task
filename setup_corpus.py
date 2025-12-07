"""
Setup script to create the corpus folder and all 6 speech files.
Run this first before running main.py or evaluation.py
"""

import os

# Create corpus directory
CORPUS_DIR = "./corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)

# Document contents from assignment
DOCUMENTS = {
    "speech1.txt": """The real remedy is to destroy the belief in the sanctity of the shastras. How do you expect to succeed if you allow the shastras to continue to be held as sacred and infallible? You must take a stand against the scriptures. Either you must stop the practice of caste or you must stop believing in the shastras. You cannot have both. The problem of caste is not a problem of social reform. It is a problem of overthrowing the authority of the shastras. So long as people believe in the sanctity of the shastras, they will never be able to get rid of caste. The work of social reform is like the work of a gardener who is constantly pruning the leaves and branches of a tree without ever attacking the roots. The real enemy is the belief in the shastras.

What is your ideal society? My ideal society would be based on liberty, equality, and fraternity. I do not want to be the slave of tradition. I do not want to be the slave of others. I want to be free. I want to be equal. I want to be fraternal. This is my ideal. And this ideal can be realized only through the abolition of caste.""",

    "speech2.txt": """The Buddha's Dhamma is the only religion which the world can have if the world is to become a society of free and equal men. The Dhamma is social. It is a code of moral conduct for society. The center of Dhamma is man and the relationship between man and man. What is Dhamma? The Dhamma is to avoid evil, to do good, and to purify the mind. This is the teaching of all Buddhas.

The Buddha was against ritualism. He condemned the performance of sacrifices. He rejected the authority of the Vedas. He denied the efficacy of prayers. He discarded the usefulness of gods. He emphasized the law of Karma. He taught that every man has the power to shape his own destiny. He made man the master of his own fate.""",

    "speech3.txt": """The constitutional remedies for the protection of minorities are of two kinds. First, fundamental rights which are justiciable. Second, directive principles which are not justiciable. The fundamental rights must include the right to equality, the right to freedom, the right against exploitation, and the right to constitutional remedies. These rights are necessary to protect the minorities against the tyranny of the majority.

The Constitution is not a mere lawyers' document. It is a vehicle of life, and its spirit is always the spirit of age. The Constitution must provide for economic democracy as well as political democracy. Political democracy cannot last unless there lies at the base of it social democracy. What does social democracy mean? It means a way of life which recognizes liberty, equality, and fraternity as the principles of life.""",

    "speech4.txt": """I was born in the untouchable community. I have faced the stigma of untouchability from my childhood. I know what it means to be an untouchable. I know the humiliation, the insults, the injustices. I remember how we were not allowed to drink water from the public well. I remember how we were not allowed to enter the temple. I remember how we were forced to live outside the village.

Education is the key to liberation. Without education, the untouchables can never achieve their rightful place in society. I struggled for education against all odds. I went to school sitting outside the classroom. I studied under street lights when I had no money for kerosene. But I was determined to get educated. Education gave me the strength to fight against injustice.""",

    "speech5.txt": """The Hindu-Muslim problem is fundamentally a problem of nationalities. The Hindus and Muslims are not two communities but two nations. They have different religions, different cultures, different languages, and different historical backgrounds. The Muslims have a consciousness of being a separate nation. They want to have their own homeland, their own national state.

The creation of Pakistan is inevitable. The Muslims cannot live under Hindu domination. They must have a separate state where they can develop according to their own culture and civilization. The unity of India is artificial. It was created by the British. Before the British, India was never united. The partition of India is the only solution to the Hindu-Muslim problem.""",

    "speech6.txt": """The untouchables are not a caste. They are a class. They are the broken men of Indian society. They were the original inhabitants of India who were defeated and subjugated by the invading Aryans. The untouchables were forced to live outside the village. They were denied the right to education. They were denied the right to property. They were condemned to do menial work.

The solution to the problem of untouchability lies in the annihilation of caste. The untouchables must fight for their rights. They must organize themselves. They must educate their children. They must enter the government services. They must use political power to protect their interests. The emancipation of the untouchables is impossible without political power."""
}

print("Creating corpus files...")
print("=" * 50)

for filename, content in DOCUMENTS.items():
    filepath = os.path.join(CORPUS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")

print("=" * 50)
print(f"✓ All {len(DOCUMENTS)} corpus files created successfully!")
print(f"✓ Location: {os.path.abspath(CORPUS_DIR)}")
print("\nYou can now run:")
print("  python main.py          - For interactive Q&A")
print("  python evaluation.py    - For comprehensive evaluation")