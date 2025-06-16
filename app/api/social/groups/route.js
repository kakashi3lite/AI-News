import { NextResponse } from 'next/server';

// GET /api/social/groups
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const userId = searchParams.get('userId');
    const category = searchParams.get('category');
    const limit = parseInt(searchParams.get('limit')) || 20;
    
    if (action === 'discover') {
      // Return discoverable groups
      const discoverableGroups = [
        {
          id: 'ai-enthusiasts',
          name: 'AI Enthusiasts',
          description: 'Discussing the latest in artificial intelligence and machine learning',
          memberCount: 1247,
          category: 'Technology',
          isPublic: true,
          avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=ai',
          tags: ['AI', 'ML', 'Technology'],
          activityLevel: 'high',
          recentActivity: new Date(Date.now() - 1000 * 60 * 30).toISOString()
        },
        {
          id: 'news-analysts',
          name: 'News Analysts',
          description: 'Professional news analysis and media literacy discussions',
          memberCount: 892,
          category: 'News',
          isPublic: true,
          avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=news',
          tags: ['News', 'Analysis', 'Media'],
          activityLevel: 'medium',
          recentActivity: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString()
        },
        {
          id: 'tech-startups',
          name: 'Tech Startups',
          description: 'Startup founders and entrepreneurs in tech',
          memberCount: 634,
          category: 'Business',
          isPublic: false,
          avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=startup',
          tags: ['Startup', 'Business', 'Tech'],
          activityLevel: 'high',
          recentActivity: new Date(Date.now() - 1000 * 60 * 15).toISOString()
        },
        {
          id: 'data-science',
          name: 'Data Science Hub',
          description: 'Data scientists sharing insights and methodologies',
          memberCount: 1089,
          category: 'Technology',
          isPublic: true,
          avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=data',
          tags: ['Data Science', 'Analytics', 'Python'],
          activityLevel: 'medium',
          recentActivity: new Date(Date.now() - 1000 * 60 * 60).toISOString()
        }
      ];
      
      let filteredGroups = discoverableGroups;
      if (category) {
        filteredGroups = filteredGroups.filter(g => 
          g.category.toLowerCase() === category.toLowerCase()
        );
      }
      
      return NextResponse.json({
        groups: filteredGroups.slice(0, limit),
        total: filteredGroups.length,
        categories: ['Technology', 'News', 'Business', 'Science']
      });
    }
    
    // Return user's groups
    const userGroups = [
      {
        id: 'ai-enthusiasts',
        name: 'AI Enthusiasts',
        description: 'Discussing the latest in artificial intelligence and machine learning',
        memberCount: 1247,
        role: 'member',
        joinedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
        avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=ai',
        unreadCount: 5,
        lastActivity: new Date(Date.now() - 1000 * 60 * 30).toISOString()
      },
      {
        id: 'news-analysts',
        name: 'News Analysts',
        description: 'Professional news analysis and media literacy discussions',
        memberCount: 892,
        role: 'moderator',
        joinedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 60).toISOString(),
        avatar: 'https://api.dicebear.com/7.x/shapes/svg?seed=news',
        unreadCount: 12,
        lastActivity: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString()
      }
    ];
    
    return NextResponse.json({
      groups: userGroups,
      total: userGroups.length
    });
  } catch (error) {
    console.error('[/api/social/groups] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/social/groups
export async function POST(request) {
  try {
    const { name, description, category, isPublic, tags, userId } = await request.json();
    
    if (!name || !description || !userId) {
      return NextResponse.json(
        { error: 'name, description, and userId are required' },
        { status: 400 }
      );
    }

    console.log('[/api/social/groups] Creating group:', name);
    
    // Mock creating a group
    const newGroup = {
      id: `group_${Date.now()}`,
      name,
      description,
      category: category || 'General',
      isPublic: isPublic !== false,
      tags: tags || [],
      createdBy: userId,
      createdAt: new Date().toISOString(),
      memberCount: 1,
      avatar: `https://api.dicebear.com/7.x/shapes/svg?seed=${name.toLowerCase()}`,
      role: 'admin'
    };

    return NextResponse.json({
      success: true,
      group: newGroup,
      message: 'Group created successfully'
    });
  } catch (error) {
    console.error('[/api/social/groups] Error creating group:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// PATCH /api/social/groups
export async function PATCH(request) {
  try {
    const { groupId, action, userId } = await request.json();
    
    if (!groupId || !action || !userId) {
      return NextResponse.json(
        { error: 'groupId, action, and userId are required' },
        { status: 400 }
      );
    }

    const validActions = ['join', 'leave', 'request', 'approve', 'reject'];
    if (!validActions.includes(action)) {
      return NextResponse.json(
        { error: 'Invalid action' },
        { status: 400 }
      );
    }

    console.log(`[/api/social/groups] ${action} group:`, groupId);
    
    let message = '';
    let newRole = null;
    
    switch (action) {
      case 'join':
        message = 'Successfully joined the group';
        newRole = 'member';
        break;
      case 'leave':
        message = 'Successfully left the group';
        newRole = null;
        break;
      case 'request':
        message = 'Join request sent';
        newRole = 'pending';
        break;
      case 'approve':
        message = 'Join request approved';
        newRole = 'member';
        break;
      case 'reject':
        message = 'Join request rejected';
        newRole = null;
        break;
    }

    return NextResponse.json({
      success: true,
      groupId,
      action,
      role: newRole,
      message,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('[/api/social/groups] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}