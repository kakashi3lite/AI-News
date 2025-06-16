import { NextResponse } from 'next/server';

// GET /api/social/notifications/settings
export async function GET() {
  try {
    // Mock notification settings
    const settings = {
      email: {
        comments: true,
        follows: true,
        shares: false,
        mentions: true,
        weeklyDigest: true,
        systemUpdates: false
      },
      push: {
        comments: true,
        follows: false,
        shares: true,
        mentions: true,
        weeklyDigest: false,
        systemUpdates: true
      },
      inApp: {
        comments: true,
        follows: true,
        shares: true,
        mentions: true,
        weeklyDigest: true,
        systemUpdates: true
      },
      frequency: {
        immediate: ['mentions', 'comments'],
        hourly: ['follows', 'shares'],
        daily: ['weeklyDigest'],
        weekly: ['systemUpdates']
      },
      quietHours: {
        enabled: true,
        start: '22:00',
        end: '08:00',
        timezone: 'UTC'
      }
    };

    return NextResponse.json(settings);
  } catch (error) {
    console.error('[/api/social/notifications/settings] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/social/notifications/settings
export async function POST(request) {
  try {
    const newSettings = await request.json();
    
    console.log('[/api/social/notifications/settings] Updating settings:', newSettings);
    
    // Mock saving settings
    const updatedSettings = {
      ...newSettings,
      lastUpdated: new Date().toISOString()
    };

    return NextResponse.json({
      success: true,
      settings: updatedSettings,
      message: 'Notification settings updated successfully'
    });
  } catch (error) {
    console.error('[/api/social/notifications/settings] Error updating settings:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}