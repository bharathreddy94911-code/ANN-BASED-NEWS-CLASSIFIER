import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getDatasetInfo = async () => {
  const response = await api.get('/dataset/info');
  return response.data;
};

export const getDatasetDistribution = async () => {
  const response = await api.get('/dataset/distribution');
  return response.data;
};

export const getDatasetExamples = async (split = 'train', page = 1, pageSize = 10) => {
  const response = await api.get('/dataset/examples?split=' + split + '&page=' + page + '&page_size=' + pageSize);
  return response.data;
};

export const predictNews = async (text) => {
  const response = await api.post('/predict', { text });
  return response.data;
};

export const startTraining = async (params) => {
  const response = await api.post('/train', params);
  return response.data;
};

export const getTrainingStatus = async (jobId) => {
  const response = await api.get('/train/status/' + jobId);
  return response.data;
};

export const startExperiment = async (params) => {
  const response = await api.post('/experiments', params);
  return response.data;
};

export const getExperiments = async () => {
  const response = await api.get('/experiments');
  return response.data;
};

export const getLatestEvaluation = async () => {
  const response = await api.get('/evaluate');
  return response.data;
};

export const getConfusionMatrix = async () => {
  const response = await api.get('/evaluate/confusion-matrix');
  return response.data;
};

export const getClassificationReport = async () => {
  const response = await api.get('/evaluate/report');
  return response.data;
};

export const getMisclassifiedExamples = async (page = 1, pageSize = 10) => {
  const response = await api.get('/evaluate/errors?page=' + page + '&page_size=' + pageSize);
  return response.data;
};

export const getExportUrl = () => API_BASE_URL + '/results/export';

export default api;
