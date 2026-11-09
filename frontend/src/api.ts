export const labels = ['no_issue', 'low', 'medium', 'high'] as const;
export type Label = typeof labels[number];
export const labelNames: Record<Label, string> = {
  no_issue: 'Нет проблемы', low: 'Низкая', medium: 'Средняя', high: 'Высокая',
};
export interface Health { status: string; model_loaded: boolean; model_version: string | null; training: boolean }
export interface Prediction { label: Label; criticality: number | null; confidence: number | null }
export interface Feedback { action: string; pending_count: number; dataset_version: string }
export interface ClassMetrics { precision: number; recall: number; 'f1-score': number; support: number }
export interface Metrics {
  accuracy: number; precision_macro: number; recall_macro: number; macro_f1: number;
  rows: number; per_class: Record<Label, ClassMetrics>; confusion_matrix: number[][]; label_order: Label[];
}
export interface Model {
  model_version: string; active: boolean; created_at: string | null; dataset_version: string | null;
  metrics: Record<string, Metrics> | null;
}
export interface PredictionRecord { id: string; text: string; label: Label; predicted_label: Label; confidence: number; correct: boolean }
export interface TrainingResult {
  status: string; model_version: string; dataset_version: string; duration_seconds: number;
  split_sizes: Record<string, number>; metrics: Record<string, Metrics>; predictions: Record<string, PredictionRecord[]>;
}

const base = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');

export async function request<T>(path: string, options?: RequestInit, allowUnavailable = false): Promise<T> {
  let response: Response;
  try {
    response = await fetch(base + path, {
      signal: path === '/train' ? undefined : AbortSignal.timeout(15000), ...options,
    });
  } catch {
    throw new Error('Не удалось связаться с backend. Проверьте, что сервер запущен на порту 8000.');
  }
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`Backend вернул некорректный ответ (${response.status}).`); }
  if (!response.ok && !(allowUnavailable && response.status === 503 && typeof data.model_loaded === 'boolean')) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((item: { loc?: string[]; msg?: string }) => `${item.loc?.slice(1).join('.') || 'Запрос'}: ${item.msg}`).join('; ')
      : typeof detail === 'string' ? detail : `Ошибка запроса (${response.status})`;
    throw new Error(message);
  }
  return data as T;
}

export function post<T>(path: string, data: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}
