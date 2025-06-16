import { NextResponse } from 'next/server';

// GET /api/social/notifications/[notificationId]
export async function GET(request, { params }) {
  try {
    const { notificationId } = params;
    
    // Mock notification data
    const notification = {
      id: notificationId,
      type: 'comment',
      title: 'New comment on your article',
      message: 'John Doe commented on "AI Trends in 2024"',
      timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      read: false,
      actionUrl: '/article/ai-trends-2024#comment-123',
      avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=john',
      priority: 'medium',
      metadata: {
        articleId: 'ai-trends-2024',
        commentId: '123',
        userId: 'john-doe'
      }
    };

    return NextResponse.json(notification);
  } catch (error) {
    console.error(`[/api/social/notifications/${params.notificationId}] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// PATCH /api/social/notifications/[notificationId]/read
export async function PATCH(request, { params }) {
  try {
    const { notificationId } = params;
    const { read = true } = await request.json();
    
    console.log(`[/api/social/notifications/${notificationId}] Marking as ${read ? 'read' : 'unread'}`);
    
    // Mock updating notification
    const updatedNotification = {
      id: notificationId,
      read,
      readAt: read ? new Date().toISOString() : null
    };

    return NextResponse.json({
      success: true,
      notification: updatedNotification,
      message: `Notification marked as ${read ? 'read' : 'unread'}`
    });
  } catch (error) {
    console.error(`[/api/social/notifications/${params.notificationId}] Error updating:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// DELETE /api/social/notifications/[notificationId]
export async function DELETE(request, { params }) {
  try {
    const { notificationId } = params;
    
    console.log(`[/api/social/notifications/${notificationId}] Deleting notification`);
    
    // Mock deleting notification
    return NextResponse.json({
      success: true,
      message: 'Notification deleted successfully',
      deletedId: notificationId
    });
  } catch (error) {
    console.error(`[/api/social/notifications/${params.notificationId}] Error deleting:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}