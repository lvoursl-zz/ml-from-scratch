import numpy as np

MAX_LOSS = 2.0


class WeightedApproxRankingMatrixFactorization(object):
    def __init__(
        self, users_num, items_num, 
        embedding_size, learning_rate=0.05, max_sampled=10,
        random_seed=257
    ):
        super(WeightedApproxRankingMatrixFactorization, self).__init__()
        np.random.seed(random_seed)
        self.user_embeddings = np.random.random(size=(users_num, embedding_size))
        self.item_embeddings = np.random.random(size=(items_num, embedding_size))
        
        # (self.random_state.rand(no_item_features, no_components) - 0.5) / no_components
        
        self.learning_rate = learning_rate
        self.max_sampled = max_sampled
        
        self.users_indexes = [uid for uid in range(users_num)]
        self.items_indexes = [iid for iid in range(items_num)]
        self.users_num = users_num
        self.items_num = items_num
        self.random_seed = random_seed
        
        self.users_positives_indexes = []
        self.positives_initialized = False

    def _init_positives(self, interaction_matrix):
        for user_index in self.users_indexes:
            self.users_positives_indexes.append(np.where(interaction_matrix[user_index] == 1)[0])
        self.positives_initialized = True
        
    def fit(self, interaction_matrix):
        if not self.positives_initialized:
            self._init_positives(interaction_matrix)
            
        for user_index in self.users_indexes:
            positive_item = np.random.choice(self.users_positives_indexes[user_index])
            
            sampled = 0
            while sampled <= self.max_sampled:
                sampled += 1
                sampled_item = np.random.choice(self.items_indexes)
                
                if interaction_matrix[user_index, sampled_item] == 1:
                    continue
                else:
                    if self.score(user_index, sampled_item) > self.score(user_index, positive_item) - 1:
                        loss_coef = np.log(np.max([1.0, np.floor((self.items_num - 1) / sampled)]))
                        if loss_coef > MAX_LOSS:
                            loss_coef = MAX_LOSS
                        loss_coef *= self.learning_rate 
                        self.user_embeddings[user_index] -= loss_coef * (
                            self.item_embeddings[sampled_item] - self.item_embeddings[positive_item]
                        )
                        self.item_embeddings[positive_item] -= loss_coef * (- self.user_embeddings[user_index]) 
                        self.item_embeddings[sampled_item] -= loss_coef * (self.user_embeddings[user_index])
                        break
            
    def score(self, user_index, item_index):
        return self.user_embeddings[user_index] @ self.item_embeddings[item_index]
    
    def score_user(self, user_index):
        return np.array([self.score(user_index, item_index) for item_index in self.items_indexes])
