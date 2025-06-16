import { NextResponse } from 'next/server';

// GET /api/social/notifications
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit')) || 20;
    const offset = parseInt(searchParams.get('offset')) || 0;
    const unreadOnly = searchParams.get('unread') === 'true';

    // Mock notifications data
    const allNotifications = [
      {
        id: '1',
        type: 'comment',
        title: 'New comment on your article',
        message: 'John Doe commented on "AI Trends in 2024"',
        timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(), // 30 min ago
        read: false,
        actionUrl: '/article/ai-trends-2024#comment-123',
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=john',
        priority: 'medium'
      },
      {
        id: '2',
        type: 'follow',
        title: 'New follower',
        message: 'Sarah Wilson started following you',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
        read: true,
        actionUrl: '/profile/sarah-wilson',
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=sarah',
        priority: 'low'
      },
      {
        id: '3',
        type: 'share',
        title: 'Article shared',
        message: 'Your article was shared 5 times in the last hour',
        timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(), // 1 hour ago
        read: false,
        actionUrl: '/analytics/shares',
        priority: 'high'
      },
      {
        id: '4',
        type: 'system',
        title: 'Weekly summary ready',
        message: 'Your weekly engagement summary is now available',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // 1 day ago
        read: true,
        actionUrl: '/analytics/weekly',
        priority: 'medium'
      },
      {
        id: '5',
        type: 'reaction',
        title: 'New reactions',
        message: '12 people reacted to your latest post',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 6).toISOString(), // 6 hours ago
        read: false,
        actionUrl: '/post/latest#reactions',
        priority: 'medium'
      }
    ];

    let notifications = allNotifications;
    if (unreadOnly) {
      notifications = notifications.filter(n => !n.read);
    }

    const paginatedNotifications = notifications.slice(offset, offset + limit);

    const response = {
      notifications: paginatedNotifications,
      total: notifications.length,
      unreadCount: allNotifications.filter(n => !n.read).length,
      hasMore: offset + limit < notifications.length
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('[/api/social/notifications] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/social/notifications (for creating notifications)
export async function POST(request) {
  try {
    const notification = await request.json();
    
    // Mock creating a notification
    const newNotification = {
      id: Date.now().toString(),
      ...notification,
      timestamp: new Date().toISOString(),
      read: false
    };

    console.log('[/api/social/notifications] Created notification:', newNotification.id);
    
    return NextResponse.json({
      success: true,
      notification: newNotification
    });
  } catch (error) {
    console.error('[/api/social/notifications] Error creating notification:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}