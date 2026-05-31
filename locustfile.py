from locust import HttpUser, task, between

class MLAPIUser(HttpUser):
    wait_time = between(1,2)

    @task(3)
    def health_check(self):
        self.client.get('/health')
    
    @task(1)
    def predict(self):
        self.client.get('/predict?text=love+load+test+this')