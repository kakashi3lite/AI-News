import { NextResponse } from 'next/server';

// POST /api/social/notifications/read-all
export async function POST(request) {
  try {
    const { before } = await request.json();
    
    console.log('[/api/social/notifications/read-all] Marking all notifications as read', before ? `before ${before}` : '');
    
    // Mock marking all notifications as read
    const timestamp = new Date().toISOString();
    const affectedCount = Math.floor(Math.random() * 10) + 1; // 1-10 notifications
    
    return NextResponse.json({
      success: true,
      message: `${affectedCount} notifications marked as read`,
      affectedCount,
      timestamp,
      before: before || null
    });
  } catch (error) {
    console.error('[/api/social/notifications/read-all] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}