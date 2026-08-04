#!/usr/bin/env node
/**
 * Quick API Key Setup - User-Friendly Version
 * Simple interface for non-technical users
 */

const EnhancedSecureKeyAgent = require('./enhanced-secure-key-agent');

class QuickSetupAgent extends EnhancedSecureKeyAgent {
  constructor() {
    super();
    this.quickMode = true;
  }

  /**
   * Quick setup workflow for users
   */
  async quickSetup() {
    try {
      console.log('\n🚀 Quick API Key Setup for AI News Dashboard');
      console.log('=' .repeat(55));
      console.log('We\'ll help you set up API keys in 3 simple steps:');
      console.log('1. 🔑 OpenAI (Required) - For AI summarization');
      console.log('2. 🔍 Google (Required) - For news search');
      console.log('3. 📰 NewsAPI (Optional) - For additional sources');
      console.log('=' .repeat(55));

      await this.ensureGitSafety();

      // Quick interactive setup
      const setupChoice = await this.askSecureQuestion(
        '\nChoose setup method:\n' +
        '1. 🎯 Quick setup (guided)\n' +
        '2. 🔧 Advanced setup (all options)\n' +
        '3. 📋 Manual setup (show instructions)\n' +
        'Choice (1/2/3): '
      );

      switch (setupChoice.trim()) {
        case '1':
          await this.guidedSetup();
          break;
        case '2':
          await this.setupKeys(); // Full setup from parent class
          break;
        case '3':
          this.showManualSetupGuide();
          break;
        default:
          console.log('❌ Invalid choice. Using guided setup...');
          await this.guidedSetup();
      }

    } catch (error) {
      console.error('\n❌ Setup failed:', error.message);
      this.showQuickTroubleshooting();
    } finally {
      this.rl.close();
    }
  }

  /**
   * Guided setup for essential keys only
   */
  async guidedSetup() {
    console.log('\n🎯 Guided Setup - Essential Keys Only\n');

    const essentialKeys = this.requiredKeys.filter(key => key.required);
    
    for (const keyConfig of essentialKeys) {
      const hasKey = await this.askSecureQuestion(
        `Do you have a ${keyConfig.service} API key? (y/n): `
      );

      if (hasKey.toLowerCase() === 'y') {
        await this.setupSingleKey(keyConfig);
      } else {
        console.log(`\n📖 Get your ${keyConfig.service} key:`);
        console.log(`   🌐 Visit: ${keyConfig.getUrl}`);
        console.log(`   📝 ${keyConfig.description}`);
        
        const proceed = await this.askSecureQuestion('\nReady to enter your key? (y/skip): ');
        if (proceed.toLowerCase() === 'y') {
          await this.setupSingleKey(keyConfig);
        } else {
          console.log(`⏭️  Skipping ${keyConfig.service} - you can add it later`);
        }
      }
    }

    await this.validateAllKeys();
    await this.performSecurityCheck();
    
    console.log('\n🎉 Quick Setup Complete!');
    console.log('✅ Essential API keys configured');
    console.log('✅ Secure storage in .env.local');
    console.log('\n🚀 Ready to start: npm run dev');
  }

  /**
   * Show manual setup guide
   */
  showManualSetupGuide() {
    console.log('\n📋 Manual Setup Guide');
    console.log('=' .repeat(30));
    console.log('\n1. Create .env.local in project root');
    console.log('2. Add your API keys in this format:\n');
    
    this.requiredKeys.forEach(key => {
      console.log(`   ${key.name}=your_actual_key_here`);
    });
    
    console.log('\n3. Get your keys from:');
    this.requiredKeys.forEach(key => {
      console.log(`   • ${key.service}: ${key.getUrl}`);
    });
    
    console.log('\n4. Test your setup:');
    console.log('   npm run validate:keys');
    
    console.log('\n💡 Tips:');
    console.log('   • .env.local is automatically ignored by git');
    console.log('   • Never commit API keys to version control');
    console.log('   • Use the guided setup if you need help');
    
    this.log('Displayed manual setup guide');
  }

  /**
   * Show quick troubleshooting
   */
  showQuickTroubleshooting() {
    console.log('\n🔧 Quick Troubleshooting');
    console.log('=' .repeat(25));
    console.log('\n❓ Common Issues:');
    console.log('   • Invalid key format → Check the API provider docs');
    console.log('   • API test failed → Check your internet connection');
    console.log('   • Permission denied → Run as administrator (Windows)');
    console.log('\n🆘 Need Help?');
    console.log('   • Run: npm run setup:keys (full setup)');
    console.log('   • Run: npm run validate:keys (test existing keys)');
    console.log('   • Check logs: logs/secure-key-agent.log');
  }
}

// Export for use in other modules
module.exports = QuickSetupAgent;

// Main execution
if (require.main === module) {
  const agent = new QuickSetupAgent();
  agent.quickSetup().catch(error => {
    console.error('❌ Fatal error:', error.message);
    process.exit(1);
  });
}
