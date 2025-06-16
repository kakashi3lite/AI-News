import { NextResponse } from 'next/server';

// GET /api/social/reactions
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const articleId = searchParams.get('articleId');
    const userId = searchParams.get('userId');
    
    if (!articleId) {
      return NextResponse.json(
        { error: 'articleId is required' },
        { status: 400 }
      );
    }

    // Mock reactions data
    const reactions = {
      articleId,
      reactions: {
        like: {
          count: Math.floor(Math.random() * 100) + 10,
          users: ['user1', 'user2', 'user3']
        },
        love: {
          count: Math.floor(Math.random() * 50) + 5,
          users: ['user4', 'user5']
        },
        laugh: {
          count: Math.floor(Math.random() * 30) + 2,
          users: ['user6']
        },
        wow: {
          count: Math.floor(Math.random() * 20) + 1,
          users: []
        },
        sad: {
          count: Math.floor(Math.random() * 10),
          users: []
        },
        angry: {
          count: Math.floor(Math.random() * 5),
          users: []
        }
      },
      userReaction: userId ? (Math.random() > 0.7 ? 'like' : null) : null,
      totalCount: 0
    };

    // Calculate total count
    reactions.totalCount = Object.values(reactions.reactions)
      .reduce((sum, reaction) => sum + reaction.count, 0);

    return NextResponse.json(reactions);
  } catch (error) {
    console.error('[/api/social/reactions] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/social/reactions
export async function POST(request) {
  try {
    const { articleId, userId, reaction, action } = await request.json();
    
    if (!articleId || !userId || !reaction) {
      return NextResponse.json(
        { error: 'articleId, userId, and reaction are required' },
        { status: 400 }
      );
    }

    const validReactions = ['like', 'love', 'laugh', 'wow', 'sad', 'angry'];
    if (!validReactions.includes(reaction)) {
      return NextResponse.json(
        { error: 'Invalid reaction type' },
        { status: 400 }
      );
    }

    console.log(`[/api/social/reactions] ${action || 'add'} reaction:`, { articleId, userId, reaction });
    
    // Mock reaction update
    const isAdd = action !== 'remove';
    const newCount = Math.floor(Math.random() * 100) + (isAdd ? 1 : 0);
    
    const response = {
      success: true,
      articleId,
      reaction,
      action: isAdd ? 'added' : 'removed',
      newCount,
      userReaction: isAdd ? reaction : null,
      timestamp: new Date().toISOString()
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('[/api/social/reactions] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// DELETE /api/social/reactions
export async function DELETE(request) {
  try {
    const { searchParams } = new URL(request.url);
    const articleId = searchParams.get('articleId');
    const userId = searchParams.get('userId');
    
    if (!articleId || !userId) {
      return NextResponse.json(
        { error: 'articleId and userId are required' },
        { status: 400 }
      );
    }

    console.log(`[/api/social/reactions] Removing reaction:`, { articleId, userId });
    
    return NextResponse.json({
      success: true,
      message: 'Reaction removed',
      articleId,
      userId,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('[/api/social/reactions] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}