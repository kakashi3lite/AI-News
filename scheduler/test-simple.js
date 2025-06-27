const yaml = require('js-yaml');
const fs = require('fs');

try {
  console.log('Loading configuration...');
  const configFile = fs.readFileSync('./jobs.yaml', 'utf8');
  const config = yaml.load(configFile);
  console.log('Configuration loaded successfully!');
  console.log('Jobs found:', Object.keys(config.jobs || {}).length);
} catch (error) {
  console.error('Error:', error.message);
  console.error('Stack:', error.stack);
}