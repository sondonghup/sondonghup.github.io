# Acer's home

개발과 사는 이야기를 적는 블로그입니다. 주소는 <https://sondonghup.github.io/> 입니다.

## 글 올리는 법

1. `_posts` 폴더에 `2026-11-20-글-제목.md` 처럼 `날짜-제목.md` 이름으로 파일을 만든다.
2. 파일 맨 위에 머리말을 적는다. `categories` 는 `dev`(개발) 또는 `life`(사는 이야기) 중 하나다.

   ```yaml
   ---
   title: 글 제목
   categories: dev
   tags: [python]
   ---
   ```

3. 머리말 아래에 마크다운으로 본문을 쓴다.
4. `master` 브랜치에 올리면 몇 분 안에 사이트에 반영된다.

## 구성

- `_config.yml`: 사이트 제목, 메뉴, 하단 링크 같은 설정
- `dev.html`, `life.html`: 갈래별 글 목록 페이지
- `aboutme.md`: 소개 페이지
- `_layouts`, `_includes`, `assets`: 테마 파일

테마는 [Beautiful Jekyll](https://beautifuljekyll.com) 을 씁니다.
