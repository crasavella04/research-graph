import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { labels } from './api';
import type { Metrics, TrainingResult } from './api';

const metrics: Metrics = {
  accuracy: 1, precision_macro: 1, recall_macro: 1, macro_f1: 1, rows: 4,
  per_class: Object.fromEntries(labels.map(label => [label, { precision: 1, recall: 1, 'f1-score': 1, support: 1 }])) as Metrics['per_class'],
  label_order: [...labels], confusion_matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
};
const trained: TrainingResult = {
  status: 'completed', model_version: 'model-new', dataset_version: 'dataset-new', duration_seconds: 2,
  split_sizes: { train: 12, validation: 4, test: 4 }, metrics: { test: metrics, validation: metrics },
  predictions: { test: [], validation: [] },
};
let modelLoaded = true;
let version = 'baseline-v1';
let doTrain: () => Promise<Response>;
const response = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status });

beforeEach(() => {
  modelLoaded = true; version = 'baseline-v1';
  doTrain = async () => { version = 'model-new'; return response(trained, 201); };
  vi.stubGlobal('fetch', vi.fn(async (input: string, options?: RequestInit) => {
    if (input.endsWith('/health')) return response({ status: modelLoaded ? 'ok' : 'unavailable', model_loaded: modelLoaded, model_version: modelLoaded ? version : null, training: false }, modelLoaded ? 200 : 503);
    if (input.endsWith('/models')) return response(modelLoaded ? [{ model_version: version, active: true, created_at: null, dataset_version: null, metrics: null }] : []);
    if (input.endsWith('/predict')) return response({ label: 'high', criticality: 2, confidence: 0.8 });
    if (input.endsWith('/feedback')) return response({ action: 'added', pending_count: 1, dataset_version: 'dataset-1' });
    if (input.endsWith('/train')) return doTrain();
    if (input.endsWith('/load')) { expect(options?.body).toBeInstanceOf(FormData); modelLoaded = true; return response({ status: 'loaded' }, 201); }
    throw new Error(`Unexpected request: ${input}`);
  }));
});

describe('Frontend workflows', () => {
  it('analyses text, saves a corrected label and clears a stale prediction on editing', async () => {
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Модель готова');
    await user.type(screen.getByLabelText('Текст отзыва'), 'Cannot use screen reader');
    await user.click(screen.getByRole('button', { name: /Анализировать отзыв/ }));
    await screen.findByRole('heading', { name: 'Высокая' });
    expect(screen.getByText(/Уверенность модели: 80.0%/)).toBeVisible();
    await user.selectOptions(screen.getByLabelText('Правильный класс'), 'medium');
    await user.click(screen.getByRole('button', { name: 'Сохранить для обучения' }));
    await screen.findByText(/Исправление сохранено/);
    const calls = vi.mocked(fetch).mock.calls;
    const saved = calls.find(([path]) => String(path).endsWith('/feedback'));
    expect(JSON.parse(String(saved?.[1]?.body))).toEqual({ text: 'Cannot use screen reader', label: 'medium' });
    await user.type(screen.getByLabelText('Текст отзыва'), ' changed');
    expect(screen.queryByRole('heading', { name: 'Высокая' })).not.toBeInTheDocument();
  });

  it('accepts a new training example even when no model exists', async () => {
    modelLoaded = false;
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Модель не загружена');
    await user.type(screen.getByLabelText('Текст отзыва'), 'New review');
    expect(screen.getByRole('button', { name: /Анализировать отзыв/ })).toBeDisabled();
    await user.selectOptions(screen.getByLabelText('Правильный класс'), 'low');
    await user.click(screen.getByRole('button', { name: 'Сохранить для обучения' }));
    await screen.findByText(/Исправление сохранено/);
  });

  it('locks controls while synchronous training is pending and displays the completed report', async () => {
    let finish!: (value: Response) => void;
    doTrain = () => new Promise(resolve => { finish = resolve; });
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Модель готова');
    await user.click(screen.getByRole('button', { name: 'Обучение' }));
    await user.click(screen.getByRole('button', { name: /Обучить модель/ }));
    await screen.findByText(/Подготовка данных, обучение и оценка/);
    expect(screen.getByRole('button', { name: 'Анализ отзыва' })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Обновить состояние/ })).toBeDisabled();
    version = 'model-new';
    await act(async () => finish(response(trained, 201)));
    await screen.findByRole('heading', { name: 'Результаты оценки' });
    expect(screen.getByText(/Обучение завершено/)).toBeVisible();
    expect(screen.getByRole('button', { name: 'Анализ отзыва' })).toBeEnabled();
    expect(screen.getByText('Матрица ошибок')).toBeVisible();
  });

  it('unlocks the interface after a training error', async () => {
    doTrain = async () => response({ detail: 'Not enough labeled groups' }, 422);
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Модель готова');
    await user.click(screen.getByRole('button', { name: 'Обучение' }));
    await user.click(screen.getByRole('button', { name: /Обучить модель/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Not enough labeled groups');
    expect(screen.getByRole('button', { name: 'Анализ отзыва' })).toBeEnabled();
  });

  it('uploads both initial model files as multipart', async () => {
    modelLoaded = false;
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Модель не загружена');
    await user.click(screen.getByRole('button', { name: 'Модели' }));
    await user.upload(screen.getByLabelText('model.joblib'), new File(['model'], 'model.joblib'));
    await user.upload(screen.getByLabelText('metadata.json'), new File(['{}'], 'metadata.json', { type: 'application/json' }));
    expect((screen.getByLabelText('model.joblib') as HTMLInputElement).files?.length).toBe(1);
    expect((screen.getByLabelText('metadata.json') as HTMLInputElement).files?.length).toBe(1);
    expect(screen.getByRole('button', { name: 'Загрузить модель' })).toBeEnabled();
    // jsdom's native required-file validation does not see user-event's FileList.
    fireEvent.submit((screen.getByLabelText('model.joblib') as HTMLInputElement).form!);
    await screen.findByText('Модель загружена и готова к работе.');
    const saved = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith('/load'));
    const form = saved?.[1]?.body as FormData;
    expect(form.get('model')).toBeInstanceOf(File);
    expect(form.get('metadata')).toBeInstanceOf(File);
    expect(saved?.[1]?.headers).toBeUndefined();
  });

  it('shows connection failure and supports refreshing after recovery', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error('Offline'));
    const user = userEvent.setup(); render(<App />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Не удалось связаться с backend');
    await user.click(screen.getByRole('button', { name: /Обновить состояние/ }));
    await waitFor(() => expect(screen.getByText('Модель готова')).toBeVisible());
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
