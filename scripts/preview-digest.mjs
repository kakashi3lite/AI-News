// Preview the daily digest email as HTML (no SMTP credentials required).
// Writes digest-preview.html you can open in a browser.
// Run: node --env-file=.env scripts/preview-digest.mjs
import fs from 'node:fs';
import { getDigest } from '../lib/signal.js';
import { renderDigestHtml, smtpConfigured, digestRecipients } from '../lib/email.js';

const digest = await getDigest();
const html = renderDigestHtml(digest);
fs.writeFileSync('digest-preview.html', html);

console.log('✅ digest-preview.html written');
console.log(`   themes=${digest.themes.length} stories=${digest.stories.length}`);
console.log(`   SMTP configured: ${smtpConfigured() ? 'yes' : 'no'}`);
console.log(`   recipients: ${digestRecipients().length ? digestRecipients().join(', ') : 'none set (add DIGEST_EMAILS)'}`);
console.log('   Open digest-preview.html to view the email.');
