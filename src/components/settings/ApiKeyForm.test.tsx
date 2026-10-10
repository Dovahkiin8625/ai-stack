// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ApiKeyForm from './ApiKeyForm';
import { DEFAULTS, type AppSettings } from '../../lib/tauri';

// `vi.mock` 工厂在 import 阶段即被调用，不能引用尚未定义的 `vi.fn()`。
// 用 `vi.hoisted` 让 mock 实例与工厂在同一时序可见。
const { testModelMock } = vi.hoisted(() => ({ testModelMock: vi.fn() }));
vi.mock('../../lib/tauri', () => ({
  testModel: (...args: unknown[]) => testModelMock(...args),
  DEFAULTS: {
    apiKey: '',
    baseUrl: 'https://api.anthropic.com',
    lightweightModel: 'claude-haiku-4-5-20251001',
    performanceModel: 'claude-sonnet-4-5',
    syncBaseUrl: '',
  },
}));

// 表单字段必须三件套非空，否则 ApiKeyForm 内部的"未填完整"前置检查会拦截 mock，
// 调用方必须显式提供 baseUrl + apiKey 才能拿到"走完后端"的真实路径。
const TEST_KEY = 'sk-test-fake';
const TEST_BASE = 'https://api.anthropic.com';

function renderForm(overrides: Partial<AppSettings> = {}, onChange = vi.fn()) {
  const form: AppSettings = { ...DEFAULTS, apiKey: TEST_KEY, baseUrl: TEST_BASE, ...overrides };
  return render(<ApiKeyForm form={form} onChange={onChange} />);
}

describe('ApiKeyForm - 测试模型功能', () => {
  beforeEach(() => {
    testModelMock.mockReset();
  });

  it('每个模型字段都有「测试」按钮', () => {
    renderForm();
    expect(screen.getAllByRole('button', { name: /测试/ }).length).toBeGreaterThanOrEqual(2);
  });

  it('点击「测试」展示成功结果（绿色）', async () => {
    testModelMock.mockResolvedValue({
      model: 'claude-haiku-4-5-20251001',
      ok: true,
      message: '可用（142 ms）',
      latencyMs: 142,
    });
    renderForm();
    // 拿第一个「测试」（对应轻量模型）
    const buttons = screen.getAllByRole('button', { name: /测试/ });
    fireEvent.click(buttons[0]);
    // 等待 promise settle + 状态写入
    const ok = await screen.findByText(/可用（142 ms）/, {}, { timeout: 1500 });
    expect(ok).toBeTruthy();
    // 确认调用时带了表单当前 base_url/api_key/model
    expect(testModelMock).toHaveBeenCalledWith({
      baseUrl: TEST_BASE,
      apiKey: TEST_KEY,
      model: DEFAULTS.lightweightModel,
    });
  });

  it('点击「测试」展示失败结果（红色 + 后端错误信息）', async () => {
    testModelMock.mockResolvedValue({
      model: 'claude-haiku-4-5-20251001',
      ok: false,
      message: 'HTTP 401：invalid x-api-key',
      latencyMs: undefined,
    });
    renderForm();
    fireEvent.click(screen.getAllByRole('button', { name: /测试/ })[0]);
    // 失败必须把后端原文展示出来 —— 用户需要直观看到"key 错了" 还是 "模型不对"
    const err = await screen.findByText(/HTTP 401/);
    expect(err).toBeTruthy();
  });

  it('未填完整字段就点测试：前端拦截，不打后端', () => {
    // baseUrl 与 apiKey 为空 → 拦截；不再让请求真的发出去。
    renderForm({ baseUrl: '', apiKey: '' });
    fireEvent.click(screen.getAllByRole('button', { name: /测试/ })[0]);
    expect(testModelMock).not.toHaveBeenCalled();
    // 提示里关键信息必须能让人一眼看出要补什么
    expect(screen.getByText(/请先填写/)).toBeTruthy();
  });

  it('后端抛错（IPC 失败）显示错误', async () => {
    testModelMock.mockRejectedValue(new Error('网络异常'));
    renderForm();
    fireEvent.click(screen.getAllByRole('button', { name: /测试/ })[0]);
    // 抛错走的 catch 分支，把 e 转字符串塞 message —— 不能吞掉。
    const err = await screen.findByText(/网络异常/);
    expect(err).toBeTruthy();
  });
});
