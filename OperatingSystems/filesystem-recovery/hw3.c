#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "ext2.h"

#define BASE_OFFSET 1024
#define EXT2_BLOCK_SIZE 1024
#define SECTOR_SIZE 512
#define IMAGE "image.img"

typedef unsigned char bmap;
#define __NBITS (8 * (int) sizeof (bmap))
#define __BMELT(d) ((d) / __NBITS)
#define __BMMASK(d) ((bmap) 1 << ((d) % __NBITS))
#define BM_SET(d, set) ((set[__BMELT (d)] |= __BMMASK (d)))
#define BM_CLR(d, set) ((set[__BMELT (d)] &= ~__BMMASK (d)))
#define BM_ISSET(d, set) ((set[__BMELT (d)] & __BMMASK (d)) != 0)

unsigned int block_size = 0;
#define BLOCK_OFFSET(block) (BASE_OFFSET + (block-1)*block_size)


void recover(int fd)
{
    struct ext2_super_block super;
    struct ext2_group_desc group;
    struct ext2_inode root_inode;
    struct ext2_inode lostfound_inode;
    struct ext2_inode read_inode, tmp_inode;
    struct ext2_dir_entry ent;
    struct ext2_dir_entry *test;
    unsigned int lostfnd_idx = 0;
    unsigned int dir_offset;
    unsigned short rec_len;
    unsigned char name_len;
    int block_num;
    char filename[] = "file00";
    char recovered_filenames[100][7];
    int rec_idx=0;
    bmap *bbitmap, *ibitmap;
    int fr = 0;
    int nfr = 0;

    // read super-block
    lseek(fd, BASE_OFFSET, SEEK_SET);
    read(fd, &super, sizeof(super));
    if (super.s_magic != EXT2_SUPER_MAGIC) {
        fprintf(stderr, "Not a Ext2 filesystem\n");
        exit(1);
    }
    block_size = 1024 << super.s_log_block_size;

    // printf("Reading from image file " IMAGE ":\n"
    //       "Blocks count            : %u\n"
    //       "First non-reserved inode: %u\n",
    //       super.s_blocks_count,
    //       super.s_first_ino);

    // read group descriptor
    lseek(fd, BASE_OFFSET + block_size, SEEK_SET);
    read(fd, &group, sizeof(group));

    bbitmap = malloc(block_size);
    lseek(fd, BLOCK_OFFSET(group.bg_block_bitmap), SEEK_SET);
    read(fd, bbitmap, block_size);
    fr = 0;
    nfr = 0;
    // printf("Free block bitmap:\n");
    // for (int i = 0; i < super.s_blocks_count; i++){
    //     if (BM_ISSET(i,bbitmap)){
    //         printf("+");    // in use
    //         nfr++;
    //     }
    //     else{
    //         printf("-");    // empty
    //         fr++;
    //     }
    // }
    // printf("\n");
    // printf("Free blocks count       : %u\n"
    //       "Non-Free block count    : %u\n"
    //       , fr, nfr);
    ////////////////////////////////
    ibitmap = malloc(block_size);
    lseek(fd, BLOCK_OFFSET(group.bg_inode_bitmap), SEEK_SET);
    read(fd, ibitmap, block_size);
    fr = 0;
    nfr = 0;
    // printf("Free inodes bitmap:\n");
    // for (int i = 0; i < super.s_inodes_count; i++){
    //     if (BM_ISSET(i,ibitmap)){
    //         printf("+");    // in use
    //         nfr++;
    //     }
    //     else{
    //         printf("-");    // empty
    //         fr++;
    //     }
    // }
    // printf("\n");
    // printf("Free inodes count       : %u\n"
    //       "Non-Free inodes count    : %u\n"
    //       , fr, nfr);

    lseek(fd, BLOCK_OFFSET(group.bg_inode_table)+sizeof(struct ext2_inode), SEEK_SET);
    read(fd, &root_inode, sizeof(struct ext2_inode));
    // printf("Reading root inode\n"
    //       "Size     : %u bytes\n"
    //       "Blocks   : %u\n"
    //       "i_mode   : %hu\n",
    //       root_inode.i_size,
    //       root_inode.i_blocks,
    //       root_inode.i_mode); // in number of sectors. A disk sector is 512 bytes.
    // for(int i=0; i < 15; i++){
    //     if (i < 12)         // direct blocks
    //         printf("Block %2u : %u\n", i, root_inode.i_block[i]);
    //     else if (i == 12)     // single indirect block
    //         printf("Single   : %u\n", root_inode.i_block[i]);
    //     else if (i == 13)    // double indirect block
    //         printf("Double   : %u\n", root_inode.i_block[i]);
    //     else if (i == 14)    // triple indirect block
    //         printf("Triple   : %u\n", root_inode.i_block[i]);
    // }

    lseek(fd, BLOCK_OFFSET(group.bg_inode_table)+sizeof(struct ext2_inode)*10, SEEK_SET);
    read(fd, &lostfound_inode, sizeof(struct ext2_inode));
    // printf("Reading lostfound_inode\n"
    //       "Size     : %u bytes\n"
    //       "Blocks   : %u\n"
    //       "i_mode   : %hu\n",
    //       lostfound_inode.i_size,
    //       lostfound_inode.i_blocks,
    //       lostfound_inode.i_mode); // in number of sectors. A disk sector is 512 bytes.
    // for(int i=0; i < 15; i++){
    //     if (i < 12)         // direct blocks
    //         printf("Block %2u : %u\n", i, lostfound_inode.i_block[i]);
    //     else if (i == 12)     // single indirect block
    //         printf("Single   : %u\n", lostfound_inode.i_block[i]);
    //     else if (i == 13)    // double indirect block
    //         printf("Double   : %u\n", lostfound_inode.i_block[i]);
    //     else if (i == 14)    // triple indirect block
    //         printf("Triple   : %u\n", lostfound_inode.i_block[i]);
    // }

    int blk_cnt = 0, lostind = 1;
    unsigned int *block_ptr = malloc(block_size), *block_ptr2 = malloc(block_size), *block_ptr3 = malloc(block_size);
    for(int j=super.s_first_ino; j<super.s_inodes_count; ++j, blk_cnt = 0)
    {
        lseek(fd, BLOCK_OFFSET(group.bg_inode_table)+sizeof(struct ext2_inode)*j, SEEK_SET);
        read(fd, &read_inode, sizeof(struct ext2_inode));
        if (BM_ISSET(j,ibitmap) || read_inode.i_dtime <= 0){
            // printf("GEREKSIZ inode %d\n", j+1);    // in use
            continue;
        }
        if(filename[5] == '9')
        {
            filename[5] = 0;
            filename[4]++;
        }
        else
            filename[5]++;
        printf("%s %d %d\n", filename, read_inode.i_dtime, read_inode.i_blocks*SECTOR_SIZE/EXT2_BLOCK_SIZE);
        blk_cnt = read_inode.i_size/EXT2_BLOCK_SIZE;
        if(read_inode.i_size % EXT2_BLOCK_SIZE != 0)
            blk_cnt++;
        // printf("\n\nDTime:%u \n", read_inode.i_dtime);
        // printf("BlockCount:%u \n", blk_cnt);
        // printf("%d Reading new inode\n"
        //       "Size     : %u bytes\n"
        //       "Blocks   : %u\n"
        //       "i_mode   : %hu\n",
        //       j+1,
        //       read_inode.i_size,
        //       read_inode.i_blocks,
        //       read_inode.i_mode); // in number of sectors. A disk sector is 512 bytes.
        // for(int i=0; i < 15; i++){
        //     if (i < 12)         // direct blocks
        //         printf("Block %2u : %u\n", i, read_inode.i_block[i]);
        //     else if (i == 12)     // single indirect block
        //         printf("Single   : %u\n", read_inode.i_block[i]);
        //     else if (i == 13)    // double indirect block
        //         printf("Double   : %u\n", read_inode.i_block[i]);
        //     else if (i == 14)    // triple indirect block
        //         printf("Triple   : %u\n", read_inode.i_block[i]);
        // }
        
        for(int k=0; k < 15; ++k)
        {
            if(k<12)
            {
                if(read_inode.i_block[k] == 0)
                    continue;
                block_num = read_inode.i_block[k];
            }
            else if(k == 12)
            {
                if(read_inode.i_block[k] == 0)
                    continue;
                if(BM_ISSET(read_inode.i_block[k], bbitmap))
                {
                    blk_cnt = 0;
                    break;
                }
                lseek(fd, block_size*read_inode.i_block[k], SEEK_SET);
                read(fd, block_ptr, block_size);
                for(int tmp2=0;tmp2<block_size/sizeof(unsigned int);tmp2++)
                {
                    if(block_ptr[tmp2]==0)
                        continue;
                    block_num = block_ptr[tmp2];
                }
            }
            else if(k == 13)
            {
                if(read_inode.i_block[k] == 0)
                    continue;
                if(BM_ISSET(read_inode.i_block[k], bbitmap))
                {
                    blk_cnt = 0;
                    break;
                }
                lseek(fd, block_size*read_inode.i_block[k], SEEK_SET);
                read(fd, block_ptr, block_size);
                for(int tmp2=0;tmp2<block_size/sizeof(unsigned int);tmp2++)
                {
                    if(block_ptr[tmp2]==0)
                        continue;
                    lseek(fd, block_size*block_ptr[tmp2], SEEK_SET);
                    read(fd, block_ptr2, block_size);
                    for(int tmp3=0;tmp3<block_size/sizeof(unsigned int);tmp3++)
                    {
                        if(block_ptr2[tmp3]==0)
                            continue;
                        block_num = block_ptr2[tmp3];
                    }
                }
            }
            else if(k == 14)
            {
                if(read_inode.i_block[k] == 0)
                    continue;
                if(BM_ISSET(read_inode.i_block[k], bbitmap))
                {
                    blk_cnt = 0;
                    break;
                }
                lseek(fd, block_size*read_inode.i_block[k], SEEK_SET);
                read(fd, block_ptr, block_size);
                for(int tmp2=0;tmp2<block_size/sizeof(unsigned int);tmp2++)
                {
                    if(block_ptr[tmp2]==0)
                        continue;
                    lseek(fd, block_size*block_ptr[tmp2], SEEK_SET);
                    read(fd, block_ptr2, block_size);
                    for(int tmp3=0;tmp3<block_size/sizeof(unsigned int);tmp3++)
                    {
                        if(block_ptr2[tmp3]==0)
                            continue;
                        lseek(fd, block_size*block_ptr2[tmp3], SEEK_SET);
                        read(fd, block_ptr3, block_size);
                        for(int tmp4=0;tmp4<block_size/sizeof(unsigned int);tmp4++)
                        {
                            if(block_ptr3[tmp4]==0)
                                continue;
                            block_num = block_ptr3[tmp4];
                        }
                    }
                }
            }
            if(BM_ISSET(block_num, bbitmap))
            {
                blk_cnt = 0;
                break;
            }
            // yet another if
            // check if another inode is using any blocks in to be recovered inode
            for(int tmp=super.s_first_ino; tmp<super.s_inodes_count; ++tmp)
            {
                lseek(fd, BLOCK_OFFSET(group.bg_inode_table)+sizeof(struct ext2_inode)*tmp, SEEK_SET);
                read(fd, &tmp_inode, sizeof(struct ext2_inode));
                if (!BM_ISSET(tmp,ibitmap) && tmp_inode.i_dtime < read_inode.i_dtime){
                    continue;
                }
                if(j == tmp)
                	continue;
                for(int tmp2=0; tmp2<15; tmp2++)
                {
                    if(tmp2<12)
                    {
                        if(tmp_inode.i_block[tmp2] == 0)
                            continue;
                        if(tmp_inode.i_block[tmp2] == block_num)
                        {
                        	printf("sa");
                            blk_cnt = 0;
                            break;
                        }
                    }
                    else if(tmp2 == 12)
                    {
                        if(tmp_inode.i_block[tmp2] == 0)
                            continue;
                        lseek(fd, block_size*tmp_inode.i_block[tmp2], SEEK_SET);
                        read(fd, block_ptr, block_size);
                        
                        for(int tmp3=0;tmp3<block_size/sizeof(unsigned int);tmp3++)
                        {
                            if(block_ptr[tmp3]==0)
                                continue;
                            if(block_num == block_ptr[tmp3])
                            {
                                blk_cnt = 0;
                                break;
                            }
                        }
                    }
                    else if(tmp2 == 13)
                    {
                        if(tmp_inode.i_block[tmp2] == 0)
                            continue;
                        lseek(fd, block_size*tmp_inode.i_block[tmp2], SEEK_SET);
                        read(fd, block_ptr, block_size);
                        
                        for(int tmp3=0;tmp3<block_size/sizeof(unsigned int);tmp3++)
                        {
                            if(block_ptr[tmp3]==0)
                                continue;
                            lseek(fd, block_size*block_ptr[tmp3], SEEK_SET);
                            read(fd, block_ptr2, block_size);
                            
                            for(int tmp4=0;tmp4<block_size/sizeof(unsigned int);tmp4++)
                            {
                                if(block_ptr2[tmp4]==0)
                                    continue;
                                if(block_num == block_ptr2[tmp4])
                                {
                                    blk_cnt = 0;
                                    break;
                                }
                            }
                            if(blk_cnt == 0)
                                break;
                        }
                        if(blk_cnt == 0)
                            break;
                    }
                    else if(tmp2 == 14)
                    {
                        if(tmp_inode.i_block[tmp2] == 0)
                            continue;
                        lseek(fd, block_size*tmp_inode.i_block[tmp2], SEEK_SET);
                        read(fd, block_ptr, block_size);
                        
                        for(int tmp3=0;tmp3<block_size/sizeof(unsigned int);tmp3++)
                        {
                            if(block_ptr[tmp3]==0)
                                continue;
                            lseek(fd, block_size*block_ptr[tmp3], SEEK_SET);
                            read(fd, block_ptr2, block_size);
                            
                            for(int tmp4=0;tmp4<block_size/sizeof(unsigned int);tmp4++)
                            {
                                if(block_ptr2[tmp4]==0)
                                    continue;
                                lseek(fd, block_size*block_ptr2[tmp4], SEEK_SET);
                                read(fd, block_ptr3, block_size);
                                
                                for(int tmp5=0;tmp5<block_size/sizeof(unsigned int);tmp5++)
                                {
                                    if(block_ptr3[tmp5]==0)
                                        continue;
                                    if(block_num == block_ptr3[tmp5])
                                    {
                                        blk_cnt = 0;
                                        break;
                                    }
                                }
                                if(blk_cnt == 0)
                                    break;
                            }
                            if(blk_cnt == 0)
                                break;
                        }
                        if(blk_cnt == 0)
                            break;
                    }
                }
                if(blk_cnt == 0)
                    break;
            }
            BM_SET(read_inode.i_block[k], bbitmap);
            if(blk_cnt == 0)
                break;
        }
        if(blk_cnt == 0)
        {
            // it is set in for loop. But not recovered.
            // printf("burdan cikiyyeah");
            lseek(fd, BLOCK_OFFSET(group.bg_block_bitmap), SEEK_SET);
            read(fd, bbitmap, block_size);
            continue;
        }
        read_inode.i_dtime = 0;
        // read_inode.i_flags = 0;
        read_inode.i_links_count = 1;
        read_inode.i_mode = EXT2_S_IFREG|EXT2_S_IRUSR;
        lseek(fd, BLOCK_OFFSET(group.bg_inode_table)+sizeof(struct ext2_inode)*j, SEEK_SET);
        write(fd, &read_inode, sizeof(struct ext2_inode));

        lseek(fd, BLOCK_OFFSET(group.bg_block_bitmap), SEEK_SET);
        write(fd, bbitmap, block_size);

        BM_SET(j, ibitmap);
        lseek(fd, BLOCK_OFFSET(group.bg_inode_bitmap), SEEK_SET);
        write(fd, ibitmap, block_size);
        
        // Add to lost+found
        dir_offset = 0;
        while(1)
        {
            lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx])+dir_offset+sizeof(unsigned int), SEEK_SET);
            read(fd, &rec_len, sizeof(unsigned short));
            lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx])+dir_offset+sizeof(unsigned int)+sizeof(unsigned short), SEEK_SET);
            read(fd, &name_len, sizeof(unsigned char));
            int actual_size = sizeof(unsigned int)+sizeof(unsigned short)+sizeof(unsigned char)*2+name_len;
            while(actual_size%4!=0)
                actual_size++;
            if(actual_size == rec_len)
            {
                dir_offset += rec_len;
            }
            else
            {
                lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx])+dir_offset+sizeof(unsigned int), SEEK_SET);
                write(fd, &actual_size, sizeof(unsigned short));
                
                test = malloc(sizeof(struct ext2_dir_entry)+sizeof(char)*name_len);
                lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx])+dir_offset, SEEK_SET);
                read(fd, test, sizeof(sizeof(struct ext2_dir_entry)+sizeof(char)*6));
                dir_offset += actual_size;
                break;
            }
            if(dir_offset > 1000)
            {
                printf("\n\nHELO\n\n");
                lostfnd_idx++;
                dir_offset = 0;
            }
        }
        test = malloc(sizeof(struct ext2_dir_entry)+sizeof(char)*6);
        test->inode = j+1;
        test->rec_len = block_size - dir_offset;
        // ent.inode = j;
        // ent.rec_len = block_size - dir_offset;
        while(test->rec_len%4!=0)
            test->rec_len++;
        test->name_len = 6;
        test->file_type = EXT2_FT_REG_FILE;
        strcpy(test->name, filename);
        strcpy(recovered_filenames[rec_idx++], filename);
        lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx])+dir_offset, SEEK_SET);
        write(fd, test, test->rec_len);
        // printf("%u, %hu, %u, %u, %s\tLast write of test idx:%d\n", test->inode, test->rec_len, test->name_len, test->file_type, test->name, dir_offset);
        // printf("%u, %hu, %u, %u, %s\n", ent.inode, ent.rec_len, ent.name_len, ent.file_type, ent.name);
        
        // int tmp;
        
        // test = malloc(sizeof(struct ext2_dir_entry)+sizeof(char)*6);
        // lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx]), SEEK_SET);
        // read(fd, test, sizeof(*test));
        // printf("%u, %hu, %u, %u, %s\tOku idx:0\n", test->inode, test->rec_len, test->name_len, test->file_type, test->name);
        // tmp = test->rec_len;
        
        // test = malloc(sizeof(struct ext2_dir_entry)+sizeof(char)*6);
        // lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx])+tmp, SEEK_SET);
        // read(fd, test, sizeof(*test));
        // printf("%u, %hu, %u, %u, %s\tOku idx:%d\n", test->inode, test->rec_len, test->name_len, test->file_type, test->name, tmp);
        // tmp += test->rec_len;
        
        // test = malloc(sizeof(struct ext2_dir_entry)+sizeof(char)*12);
        // lseek(fd, BLOCK_OFFSET(lostfound_inode.i_block[lostfnd_idx])+tmp, SEEK_SET);
        // read(fd, test, sizeof(*test));
        // printf("%u, %hu, %u, %u, %sOku idx:%d\n", test->inode, test->rec_len, test->name_len, test->file_type, test->name, tmp);
    }
    printf("###");
    for(int i = 0; i < rec_idx; ++i)
    {
        printf("\n%s", recovered_filenames[i]);
    }
}


int main(int argc, char **argv)
{
    
    int fd;

    if ((fd = open(argv[1], O_RDWR)) < 0) {
        perror(argv[1]);
        exit(1);
    }


    recover(fd);


    close(fd);
    return 0;
}

