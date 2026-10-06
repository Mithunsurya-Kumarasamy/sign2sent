const API_BASE_URL = 'http://localhost:8000/api';

export const apiClient = {
  async get(endpoint) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) {
      throw new Error(`API GET Error: ${response.statusText}`);
    }
    return response.json();
  },

  async post(endpoint, data) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      throw new Error(`API POST Error: ${response.statusText}`);
    }
    return response.json();
  },

  async postForm(endpoint, formData) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`API POST Form Error: ${response.statusText}`);
    }
    return response.json();
  }
};
