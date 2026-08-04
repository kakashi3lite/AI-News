import nodemailer from 'nodemailer';

/**
 * Email digest delivery (server mode). SMTP is configured via env vars:
 *   SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, DIGEST_EMAILS
 * If SMTP is not configured, sendDigestEmail returns { skipped: true }.
 */

export function smtpConfigured() {
  return Boolean(process.env.SMTP_HOST && process.env.SMTP_USER && process.env.SMTP_PASS);
}

export function digestRecipients() {
  return String(process.env.DIGEST_EMAILS || '')
    .split(',')
    .map((e) => e.trim())
    .filter(Boolean);
}

function transport() {
  return nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT) || 587,
    secure: String(process.env.SMTP_SECURE) === 'true',
    auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
  });
}

function renderHtml(digest) {
  const theme = (t) =>
    `<div style="margin:8px 0;padding:10px 14px;background:#f1f5f9;border-left:3px solid #2563eb;border-radius:6px">
      <strong style="text-transform:capitalize">${t.name}</strong>
      <span style="color:#64748b"> — ${t.articleCount} stories
      ${t.velocity > 0 ? ` · <span style="color:#16a34a">+${t.velocity} today</span>` : ''}
      · impact ${t.impactLabel || 'n/a'} · ${t.sentimentLabel || 'neutral'}</span>
    </div>`;

  const story = (s) =>
    `<li style="margin:8px 0;line-height:1.45">
       <a href="${s.url}" style="color:#1d4ed8;font-weight:600;text-decoration:none">${s.title}</a>
       <div style="color:#64748b;font-size:12px">${s.source?.name || ''}
         · ${s.verification || 'unverified'} · impact ${s.impactScore ?? 'n/a'}
         ${s.outlook ? `· <em>${s.outlook}</em>` : ''}</div>
     </li>`;

  return `
  <div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:640px;margin:0 auto;color:#0f172a">
    <div style="background:linear-gradient(90deg,#2563eb,#4f46e5);padding:24px 28px;border-radius:12px 12px 0 0;color:#fff">
      <h1 style="margin:0;font-size:22px">Market Signal — Daily Digest</h1>
      <div style="opacity:.85;font-size:13px">${new Date(digest.generatedAt).toDateString()} · your competitive intelligence briefing</div>
    </div>
    <div style="background:#fff;padding:24px 28px;border:1px solid #e2e8f0;border-top:0;border-radius:0 0 12px 12px">
      <h2 style="font-size:16px;margin:0 0 8px">Theme pulse</h2>
      ${(digest.themes || []).map(theme).join('') || '<p>No themes yet.</p>'}
      <h2 style="font-size:16px;margin:24px 0 8px">Top stories</h2>
      <ol>${(digest.stories || []).map(story).join('') || '<li>No stories yet.</li>'}</ol>
      <p style="color:#94a3b8;font-size:12px;margin-top:24px">You're receiving this because your email is on the Market Signal digest list.</p>
    </div>
  </div>`;
}

export async function sendDigestEmail(digest) {
  if (!smtpConfigured()) {
    return { skipped: true, reason: 'SMTP not configured' };
  }
  const to = digestRecipients();
  if (to.length === 0) {
    return { skipped: true, reason: 'No DIGEST_EMAILS configured' };
  }

  const t = transport();
  await t.sendMail({
    from: process.env.SMTP_FROM || 'Market Signal <digest@marketsignal.app>',
    to: to.join(', '),
    subject: `Market Signal — Daily Digest (${new Date(digest.generatedAt).toDateString()})`,
    html: renderHtml(digest),
  });
  return { sent: true, to };
}
