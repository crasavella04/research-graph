import { useEffect, useRef, useState } from 'react';
import { labelNames, labels, post, request } from './api';
import type { Feedback, Health, Label, Metrics, Model, Prediction, PredictionRecord, TrainingResult } from './api';

const percentage = (value: number) => `${(value * 100).toFixed(1)}%`;
const criticality = { no_issue: null, low: 0, medium: 1, high: 2 };
type Tab = 'review' | 'training' | 'models';

function MetricReport({ metrics }: { metrics: Metrics }) {
  return <>
    <div className="metrics">
      {([['Accuracy', metrics.accuracy], ['Precision · macro', metrics.precision_macro],
        ['Recall · macro', metrics.recall_macro], ['Macro-F1', metrics.macro_f1]] as const).map(([name, value]) =>
        <div key={name}><span>{name}</span><strong>{percentage(value)}</strong></div>)}
    </div>
    <div className="table-wrap"><table>
      <caption>Результаты по классам</caption>
      <thead><tr><th>Класс</th><th>Precision</th><th>Recall</th><th>F1</th><th>Примеров</th></tr></thead>
      <tbody>{labels.map(label => <tr key={label}><th><span className={`tag ${label}`}>{labelNames[label]}</span></th>
        <td>{percentage(metrics.per_class[label].precision)}</td><td>{percentage(metrics.per_class[label].recall)}</td>
        <td>{percentage(metrics.per_class[label]['f1-score'])}</td><td>{metrics.per_class[label].support}</td></tr>)}</tbody>
    </table></div>
    <div className="table-wrap"><table className="matrix">
      <caption>Матрица ошибок <span>Строки — заданные классы, столбцы — предсказания</span></caption>
      <thead><tr><th>Факт / Прогноз</th>{metrics.label_order.map(label => <th key={label}>{labelNames[label]}</th>)}</tr></thead>
      <tbody>{metrics.confusion_matrix.map((row, i) => <tr key={i}><th>{labelNames[metrics.label_order[i]]}</th>
        {row.map((count, j) => <td key={j} className={i === j ? 'diagonal' : ''}>{count}</td>)}</tr>)}</tbody>
    </table></div>
  </>;
}

export default function App() {
  const [tab, setTab] = useState<Tab>('review');
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const busyRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [text, setText] = useState('');
  const [label, setLabel] = useState<Label>('no_issue');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [split, setSplit] = useState('test');
  const [seed, setSeed] = useState('42');
  const [c, setC] = useState('1');
  const [iterations, setIterations] = useState('1000');
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [metadataFile, setMetadataFile] = useState<File | null>(null);
  const [seconds, setSeconds] = useState(0);
  const locked = busy !== null || health?.training === true;

  async function refresh() {
    const [status, modelList] = await Promise.all([
      request<Health>('/health', undefined, true), request<Model[]>('/models'),
    ]);
    setHealth(status); setModels(modelList);
  }

  useEffect(() => {
    let alive = true;
    Promise.all([request<Health>('/health', undefined, true), request<Model[]>('/models')])
      .then(([status, list]) => { if (alive) { setHealth(status); setModels(list); } })
      .catch(err => { if (alive) setError(err instanceof Error ? err.message : 'Не удалось получить состояние сервиса.'); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!health?.training || busy) return;
    const timer = window.setInterval(() => { refresh().catch(err => setError(String(err))); }, 2000);
    return () => window.clearInterval(timer);
  }, [health, busy]);

  useEffect(() => {
    if (busy !== 'Обучение модели') return;
    setSeconds(0);
    const timer = window.setInterval(() => setSeconds(value => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [busy]);

  async function run(title: string, action: () => Promise<void>) {
    if (busyRef.current || health?.training) return;
    busyRef.current = true; setBusy(title); setError(null); setNotice(null);
    try { await action(); }
    catch (err) {
      setError((err instanceof Error ? err.message : 'Не удалось выполнить запрос.') +
        (title === 'Обучение модели' ? ' Если соединение прервалось, обновите состояние перед повторным запуском: обучение могло продолжиться на сервере.' : ''));
    } finally { busyRef.current = false; setBusy(null); }
  }

  function edit(record: PredictionRecord) {
    setText(record.text); setLabel(record.label);
    setPrediction({ label: record.predicted_label, criticality: criticality[record.predicted_label], confidence: record.confidence });
    setTab('review'); setNotice('Выберите правильный класс и сохраните исправление.'); setError(null);
  }

  const active = models.find(model => model.active);
  const metrics = result?.metrics[split] || active?.metrics?.[split];
  const errors = result?.predictions[split]?.filter(row => !row.correct) || [];
  const validText = text.trim().length > 0 && text.length <= 10000 && /[\p{L}\p{N}]/u.test(text);

  return <div className="shell">
    <header className="header"><a className="brand" href="#"><span className="brand-mark" aria-hidden="true">rg</span><span>research graph<small>Доступность приложений</small></span></a>
      <div className={`connection ${health?.model_loaded ? 'online' : ''}`}><span className="dot" />{health ? (health.model_loaded ? 'Модель готова' : 'Модель не загружена') : 'Нет соединения'}</div>
    </header>
    <div className="intro"><div><p className="eyebrow">РАБОЧАЯ ПАНЕЛЬ</p><h1>От отзыва — к исправлению</h1><p>Оцените критичность проблемы, уточните класс и обучите модель на новых данных.</p></div>
      <div className="version"><span>Активная модель</span><strong>{health?.model_version || 'Пока нет'}</strong></div></div>
    {error && <div className="message error" role="alert">{error}</div>}
    {notice && <div className="message success" role="status">{notice}</div>}
    <fieldset disabled={locked} className="workspace">
      <legend className="sr-only">Управление моделью</legend>
      <div className="navigation"><nav aria-label="Разделы">
        {([['review', 'Анализ отзыва'], ['training', 'Обучение'], ['models', 'Модели']] as const).map(([id, title]) =>
          <button key={id} className={tab === id ? 'tab active' : 'tab'} aria-pressed={tab === id} onClick={() => setTab(id)}>{title}</button>)}
      </nav><button className="text-button" onClick={() => run('Обновление', refresh)}>↻ Обновить состояние</button></div>

      <main>
        {tab === 'review' && <div className="columns">
          <section className="card"><div className="section-heading"><span className="step">01</span><div><h2>Текст отзыва</h2><p>Модель обучена на английском языке.</p></div></div>
            <form onSubmit={event => { event.preventDefault(); run('Анализ отзыва', async () => {
              const value = await post<Prediction>('/predict', { text }); setPrediction(value); setLabel(value.label);
            }); }}>
              <label htmlFor="review" className="sr-only">Текст отзыва</label>
              <textarea id="review" value={text} maxLength={10000} placeholder="Screen reader does not announce the login button…" rows={8}
                onChange={event => { setText(event.target.value); setPrediction(null); setNotice(null); }} />
              <div className="input-footer"><button type="button" className="text-button" onClick={() => { setText('Screen reader does not announce the login button'); setPrediction(null); }}>Вставить пример</button><span>{text.length.toLocaleString('ru-RU')} / 10 000</span></div>
              <button className="primary" type="submit" disabled={!validText || !health?.model_loaded}>Анализировать отзыв <span aria-hidden="true">↗</span></button>
            </form>
            {!health?.model_loaded && <p className="hint">Для анализа сначала обучите модель или загрузите готовые файлы во вкладке «Модели».</p>}
          </section>
          <section className="card"><div className="section-heading"><span className="step">02</span><div><h2>Результат и обратная связь</h2><p>Исправления попадут в следующее обучение.</p></div></div>
            {prediction ? <div className={`prediction ${prediction.label}`}><span className="eyebrow">КРИТИЧНОСТЬ</span><h3>{labelNames[prediction.label]}</h3><p>{prediction.confidence === null ? 'Уверенность недоступна' : `Уверенность модели: ${percentage(prediction.confidence)}`}</p></div>
              : <div className="empty"><span aria-hidden="true">◎</span><p>Здесь появится результат анализа</p><small>Можно сразу добавить свой текст с правильным классом.</small></div>}
            <form className="feedback" onSubmit={event => { event.preventDefault(); run('Сохранение исправления', async () => {
              const saved = await post<Feedback>('/feedback', { text, label });
              setNotice(`Исправление ${saved.action === 'updated' ? 'обновлено' : 'сохранено'}. Ожидают обучения: ${saved.pending_count}.`);
            }); }}>
              <label htmlFor="correct-label">Правильный класс</label>
              <select id="correct-label" value={label} onChange={event => setLabel(event.target.value as Label)}>{labels.map(value => <option key={value} value={value}>{labelNames[value]}</option>)}</select>
              <button className="secondary" disabled={!validText} type="submit">Сохранить для обучения</button>
            </form>
          </section>
        </div>}

        {tab === 'training' && <>
          <section className="card"><div className="section-heading"><span className="step">01</span><div><h2>Обучение модели</h2><p>Последний датасет и сохранённые исправления объединятся автоматически.</p></div></div>
            <form onSubmit={event => { event.preventDefault(); run('Обучение модели', async () => {
              const trained = await post<TrainingResult>('/train', { seed: Number(seed), c: Number(c), max_iter: Number(iterations) });
              setResult(trained); setSplit('test');
              setNotice(`Обучение завершено за ${trained.duration_seconds.toFixed(1)} с. Новая модель активна.`);
              try { await refresh(); } catch { setError('Модель обучена, но обновить состояние не удалось. Нажмите «Обновить состояние».'); }
            }); }}>
              <div className="training-flow"><span>Данные + feedback</span><span aria-hidden="true">→</span><span>Train / validation / test</span><span aria-hidden="true">→</span><span>Новая модель</span></div>
              <details><summary>Параметры обучения</summary><div className="parameters">
                <label>Seed<input type="number" min={0} max={4294967295} step={1} required value={seed} onChange={event => setSeed(event.target.value)} /></label>
                <label>C<input type="number" min={0.000001} step="any" required value={c} onChange={event => setC(event.target.value)} /></label>
                <label>Максимум итераций<input type="number" min={1} step={1} required value={iterations} onChange={event => setIterations(event.target.value)} /></label>
              </div></details>
              <div className="training-footer"><p className="hint">Будут сохранены новые версии данных и модели. Окно должно оставаться открытым до завершения запроса.</p><button type="submit" className="primary">Обучить модель <span aria-hidden="true">↗</span></button></div>
            </form>
          </section>
          {metrics && <section className="card report"><div className="report-heading"><div><h2>Результаты оценки</h2><p>{result ? `${result.model_version} · ${result.duration_seconds.toFixed(1)} с` : active?.model_version}</p></div><label>Выборка<select value={split} onChange={event => setSplit(event.target.value)}><option value="test">Test</option><option value="validation">Validation</option></select></label></div>
            {result && <p className="hint">Train: {result.split_sizes.train} · Validation: {result.split_sizes.validation} · Test: {result.split_sizes.test}</p>}
            <MetricReport metrics={metrics} />
            {result && <details className="errors"><summary>Ошибки классификации — {errors.length}</summary><p className="hint">Показаны первые 50 ошибок. Выберите отзыв, чтобы исправить метку.</p>
              {errors.slice(0, 50).map(row => <article className="error-row" key={row.id}><p>{row.text}</p><div><span>Задано: {labelNames[row.label]} · Прогноз: {labelNames[row.predicted_label]}</span><button className="text-button" onClick={() => edit(row)}>Исправить →</button></div></article>)}
              {errors.length === 0 && <p>На этой выборке ошибок нет.</p>}
            </details>}
          </section>}
        </>}

        {tab === 'models' && <div className="models-layout"><section className="card"><h2>Сохранённые модели</h2><p>После обучения автоматически используется последняя версия.</p>
          {models.length === 0 ? <div className="empty"><p>Сохранённых моделей пока нет</p><small>Загрузите готовую модель или запустите обучение.</small></div> : <div className="model-list">{models.map(model => <article key={model.model_version} className="model-row"><div><h3>{model.model_version}</h3><p>{model.created_at ? new Date(model.created_at).toLocaleString('ru-RU') : 'Дата неизвестна'}</p>
            {model.metrics?.test && <small>Test Macro-F1: {percentage(model.metrics.test.macro_f1)}</small>}</div><span className={`tag ${model.active ? 'no_issue' : 'archive'}`}>{model.active ? 'Активна' : 'Архив'}</span></article>)}</div>}
        </section><section className="card"><h2>Первоначальная загрузка</h2><p>Файлы из команды <code>rg ml train</code>.</p>
          {models.length > 0 ? <div className="empty"><p>Модель уже есть на диске</p><small>Чтобы создать новую версию, используйте обучение.</small></div> : <form className="upload" onSubmit={event => { event.preventDefault(); run('Загрузка модели', async () => {
            if (!modelFile || !metadataFile) return;
            const body = new FormData(); body.append('model', modelFile); body.append('metadata', metadataFile);
            await request('/load', { method: 'POST', body }); setNotice('Модель загружена и готова к работе.');
            await refresh();
          }); }}>
            <label>model.joblib<input type="file" accept=".joblib" required onChange={event => setModelFile(event.target.files?.[0] || null)} /></label>
            <label>metadata.json<input type="file" accept=".json" required onChange={event => setMetadataFile(event.target.files?.[0] || null)} /></label>
            <button className="primary" disabled={!modelFile || !metadataFile}>Загрузить модель</button>
          </form>}
        </section></div>}
      </main>
    </fieldset>
    <footer>Research Graph <span>TF-IDF + Logistic Regression · EN</span></footer>
    {locked && <div className="busy-overlay" role="status" aria-live="polite"><div className="busy-card"><div className="spinner" /><h2>{busy || 'Обучение модели'}</h2><p>{busy === 'Обучение модели' ? `Подготовка данных, обучение и оценка. Прошло ${seconds} с.` : health?.training ? 'На сервере выполняется обучение. Ожидаем завершения…' : 'Ожидаем ответ сервера…'}</p><small>Пожалуйста, не закрывайте страницу.</small></div></div>}
  </div>;
}
